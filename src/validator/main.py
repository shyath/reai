"""
Author: Eddie
"""

import argparse
import asyncio
import datasets
import time
import os
import concurrent.futures
import nltk
import traceback
from functools import partial
from typing import Any, Dict, List, Tuple

from communex.module.module import Module
from communex.module.client import ModuleClient
from communex.client import CommuneClient, Keypair
from communex.compat.key import classic_load_key
from communex._common import get_node_url

from substrateinterface import Keypair

from loguru import logger

from comx.interface import ComxInterface
from comx.client import ComxClient
from comx.miner.module import MinerModule, ScoredMinerModule
from comx.miner.registry import MinerRegistry

from config.validator import ValidatorConfig

from validator.adjust_scoring import conditional_power_scaling, normalize_scores
from validator.ats import ATS
from validator.job_description import JobDescriptionParser
from validator.resume_extract import ResumeExtractor
from validator.skills import JDSkills
from validator.io.weights import WeightIO, WeightIOInterface
from validator.io.io import IO


class Validator(Module):

    def __init__(
        self,
        key: Keypair,
        netuid: int,
        client: ComxInterface,
        weight_io: WeightIOInterface,
        interval: int,
        call_timeout: int = 20,
        use_testnet: bool = False,
    ) -> None:
        super().__init__()
        self.client = client
        self.weight_io = weight_io
        self.key = key
        self.netuid = netuid
        self.interval = interval
        self.call_timeout = call_timeout
        self.use_testnet = use_testnet
        self.uid = None
        self.queried_miners: MinerRegistry = MinerRegistry()
        self.jd_keys = JobDescriptionParser()
        self.ats = None

    def get_validator_uid(self) -> int:
        modules = self.client.get_map_modules(self.netuid, False)
        if self.key.ss58_address not in modules:
            return -1
        return modules[self.key.ss58_address]['uid']

    async def validate_step(self):
        try:
            print("Starting Asynchronous Validation Step...")
            miners = self.get_miner_modules()
            print(f"Miners: {miners}")
            print("Syncing miners...")
            new_registry = self.sync_miners(miners=miners)
            self.sync_cache(registry=new_registry)

            print("Getting next miners to query...")
            next_miners = self.next_miners(registry=new_registry)
            print("Generating job description...")
            job_description = await self.get_job_description()
            print("Retrieving resumes...")
            resumes = await self.query(miners=next_miners, job_description=job_description)
            print("Processing job description")
            scoring_data = self.process_job_description(job_description=job_description)

            print("Creating ATS object...")
            self.ats = ATS(skills_df=scoring_data['skills'], universal_skills_weights=scoring_data['universal'],
                           preferred_skills_weights=scoring_data['preferred'])

            print("Scoring miners...")
            next_miners = self.score(miners=next_miners, resumes=resumes, scoring_data=scoring_data)
            print("Cache-ing miners...")
            self.cache(miners=next_miners)

            print("Writing weights...")
            self.weight_io.write_weights(new_registry)

            if len(self.queried_miners.get_all_by_ss58()) == len(new_registry.get_all_by_ss58()):
                print("Finalizing weights...")
                uids, weights = self.update_weights()
                print("Time to vote!")
                self.vote(uids, weights)
                print("Voting Complete. Resetting Miner Registry...")
                self.queried_miners = MinerRegistry()
        except Exception as e:
            # Print the traceback
            print("Exception in 'validate_step':")
            traceback.print_exc()

    def get_miner_modules(self) -> list[MinerModule]:
        """
        Gets a list of all the miners currently registered on the subnet.

        Returns:
            A list of MinerModules representing all the miners currently
            registered to the subnet.
        """
        # Get all modules registered on subnet
        modules = self.client.get_map_modules(self.netuid, False)

        # Get subnet hyperparameters
        subnet = self.client.get_subnet_params(key=self.netuid)

        # Get maximum weight age for the subnet
        max_weight_age = subnet['max_weight_age']

        # Get the current block
        current_block = self.client.get_current_block()

        miners: list[MinerModule] = []

        # Loop through the modules and append miners to the miner list
        for m in modules.values():
            uid = m['uid']
            ss58 = m['key']
            address = m['address']

            # Always ignore yourself
            if uid == self.uid:
                continue

            last_update = m['last_update']
            reg_block = m['regblock']

            if self.is_miner(last_update, reg_block, max_weight_age, current_block):
                miners.append(MinerModule(uid=uid, ss58=ss58, address=address))

        return miners

    def is_miner(self, last_update: int, reg_block: int, max_weight_age: int, current_block: int) -> bool:
        if last_update == reg_block:
            # Its possible a validator just registered and hasn't scored miners yet.
            # This is ok because it will soon begin scoring while in its immunity period
            # resulting in this condition no longer being true.
            return True
        else:
            # Its possible a validators last_update + max_weight_age is less than current_block
            # but that means the validator is sleeping. Validator will begin to score 0's as a
            # miner unless it wakes up and starts validating again.
            if last_update + max_weight_age < current_block:
                return True

            return False

    def sync_miners(self, miners: list[MinerModule]) -> MinerRegistry:
        """
        Syncs the miners registered on the network with the persisted miners in the
        weights file. The result is returned as a MinerRegistry containing all the
        up-to-date UID-SS58 mappings. If a SS58 resulted in a changed UID, the scores
        are remembered upon updating.

        Args:
            miners: A list of MinerModules holding the current state of the network.

        Returns:
            A MinerRegistry containing the up-to-date UUID-SS58 mappings.
        """
        new_registry = MinerRegistry()
        old_registry = self.weight_io.read_weights() or MinerRegistry()

        for miner in miners:
            # Very important to get miner from the file using the ss58. This allows us
            # to remember the score of the miner incase the UID changed.
            old_miner = old_registry.get_by_ss58(miner.ss58)

            # If a miner with the same ss58 was already written to the miner stored in
            # weights file, transfer the score. Otherwise, this is a newly registered
            # miner and default to 0.
            score = 0
            if old_miner is not None:
                score = old_miner.score

            # Write the most up-to-date miner values to the new registry.
            new_registry.set(ScoredMinerModule(
                miner.uid,
                miner.ss58,
                miner.address,
                score
            ))

        return new_registry

    def sync_cache(self, registry: MinerRegistry):
        """
        Syncs the cached miners that already been queried with the contents of 
        registry. Cached miners are updated upon a UID-SS58 change, or if a miner
        deregistered and is no longer contained in the registry.

        This method should be called immediately following a call to sync_miners.

        Args:
            registry: A MinerRegistry holding the most up-to-date miners on the network.
        """
        # Ensure the registry that holds the already queried miners
        # is up to date with the network.
        cached_miners = self.queried_miners.get_all_by_ss58()
        for k, v in cached_miners.items():
            updated_miner = registry.get_by_ss58(k)

            # If the cached miner is no longer registered delete it
            # from the cached miners registry.
            if updated_miner is None:
                self.queried_miners.delete_by_ss58(k)
                continue

            # If the miners UID changes, delete it by UID so the
            # cached miner is not duplicated in the registry with
            # two different UIDs.
            if updated_miner.uid != v.uid:
                self.queried_miners.delete_by_uid(v.uid)

            # Write the up-to-date miner to the cached miner registry.
            self.queried_miners.set(ScoredMinerModule(
                uid=updated_miner.uid,
                ss58=updated_miner.ss58,
                address=updated_miner.address,
                score=updated_miner.score
            ))

    def next_miners(self, registry: MinerRegistry, count: int = 8) -> MinerRegistry:
        """
        Determines and returns the next miners that should be queried. It chooses
        up to the count amount that have not yet been queried for this voting cycle.

        Args:
            registry:
                The MinerRegistry representing the current state of the network.
            count:
                The maximum number of miners that should be returned. Defaults to 8.

        Returns:
            A MinerRegistry containing the miners that should be queried next.
        """
        counter = 0
        next_miners = MinerRegistry()
        new_registry_dict = registry.get_all_by_uid()
        for _, v in new_registry_dict.items():
            queried_miner = self.queried_miners.get_by_ss58(v.ss58)

            # If miner was already queried skip and go to the next.
            if queried_miner is not None:
                continue

            next_miners.set(v)
            counter += 1

            if counter == count:
                break

        return next_miners

    async def query(self, miners: MinerRegistry, job_description: str) -> list[dict | None]:
        """
        Queries all the miners in the MinerRegistry with the provided job description. 
        The miner responses are appended to a list in json format.

        Args:
            miners: The MinerRegistry containing the miners that will be queried.
            job_description: Dictionary containing job description info

        Returns:
            list[dict | None]:
                The list of miner responses in json format, if a query failed, it is appended
                as None.

                TODO: Format the data in a way to map uid/ss58 to each response.
        """
        get_miner_prediction = partial(self._get_miner_prediction, job_description)
        miners_dict = miners.get_all_by_uid()

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            it = executor.map(get_miner_prediction, miners_dict.values())
            miner_answers = [*it]

        print("Resumes:")
        for uid, resume in zip(miners_dict.keys(), miner_answers):
            print(f"{uid}: {resume[str(uid)]}")

        return miner_answers

    def _get_miner_prediction(self, job_description: str, miner: ScoredMinerModule) -> dict[str, Any]:

        ip, port = miner.get_split_ip_port()
        uid = miner.uid
        if ip is None or port is None or ip == 'None' or port == 'None':
            miner_prediction = None
        else:
            client = ModuleClient(host=ip, port=int(port), key=self.key)

            try:
                miner_answer = asyncio.run(
                    client.call("generate", miner.ss58, {"prompt": job_description}, timeout=self.call_timeout)
                )

                miner_prediction = miner_answer["answer"]
            except Exception as e:
                logger.error(f"Error getting miner response: {e}")
                miner_prediction = None
        miner_prediction = {str(uid): miner_prediction}
        return miner_prediction

    def score(self, miners: MinerRegistry, resumes: Dict[str, Any], scoring_data: Dict[str, Any]) -> MinerRegistry:
        """
        Takes a list of miners that will be scored, a uid mapping of resumes and job description scoring data

        Args:
            miners: The MinerRegistry containing the miners that will be scored.
            resumes: Dict containing UID as key and resume data as the value
            scoring_data: Dict containing various values extracted from the job description
        """
        miners_dict = miners.get_all_by_uid()
        uids = [u for u in miners_dict.keys()]
        print_scores = []
        for uid, v in miners_dict.items():
            resume_index = uids.index(uid)
            uid = str(uid)
            resume_data = resumes[resume_index]
            self.ats.store_resume(resume_data=resume_data)
            ats_score = self.ats.calculate_ats_score(scoring_data['jd'], uid)
            total_score = ats_score['total_score']
            v.score = total_score
            miners.set(v)
            print_score = f"{uid} - {ats_score}"
            print_scores.append(print_score)
        score_print = '\n'.join(print_scores)
        print(f"Scores:\n{score_print}")

        return miners

    def cache(self, miners: MinerRegistry):
        """
        Takes a MinerRegistry containing the queried and scored miners for
        this validator step and caches the data in queried_miners. The miners
        parameter should only contain the miners that need to be appended to
        the cache, queried_miners.

        Args:
            miners:
                The MinerRegistry containing the queried and scored miners for
                the current validator step.
        """
        miners_dict = miners.get_all_by_uid()
        for k, v in miners_dict.items():
            self.queried_miners.set(ScoredMinerModule(
                uid=k,
                ss58=v.ss58,
                address=v.address,
                score=v.score
            ))

    def update_weights(self) -> Tuple[List[int], List[int]]:
        """
        Set weights for miners based on their normalized and power scaled scores.
        """
        queried_miners = self.queried_miners.get_all_by_uid()
        uids = [u for u in queried_miners.keys()]
        weights = [w.score for w in queried_miners.values()]
        full_score_dict = {k: v for k, v in zip(uids, weights)}
        weighted_scores: dict[int: float] = {}

        abnormal_scores = full_score_dict.values()
        normal_scores = normalize_scores(abnormal_scores)
        score_dict = {uid: score for uid, score in zip(full_score_dict.keys(), normal_scores)}
        power_scaled_scores = conditional_power_scaling(score_dict)
        scores = sum(power_scaled_scores.values())

        for uid, score in power_scaled_scores.items():
            weight = score * 1000 / scores
            weighted_scores[uid] = weight

        weighted_scores = {k: v for k, v in zip(
            weighted_scores.keys(), normalize_scores(weighted_scores.values())) if v != 0}

        if self.uid is not None and str(self.uid) in weighted_scores:
            del weighted_scores[str(self.uid)]
            logger.info(f"REMOVING UID !!!!!! {self.uid}")
        else:
            logger.info("NOT REMOVING ANY UID")

        uids = list(weighted_scores.keys())
        intuids = [int(i) for i in uids]
        weights = list(weighted_scores.values())
        intweights = [int(weight * 1000) for weight in weights]

        logger.info("**********************************")
        logger.info(f"UIDS: {intuids}")
        logger.info(f"WEIGHTS TO SET: {intweights}")
        logger.info("**********************************")
        return intuids, intweights

    def validation_loop(self) -> None:
        while True:
            uid = self.get_validator_uid()
            self.uid = uid

            if self.uid != -1:
                logger.info("Begin validator step... ")
                asyncio.run(self.validate_step())
            else:
                logger.info("Validator not registered, skipping step...")

            logger.info(f"Sleeping for {self.interval} seconds... ")
            time.sleep(self.interval)

    def extract_resumes(self, miner_resumes: Dict[str, Any]) -> Dict[str, Any]:
        resume_extractor = ResumeExtractor()
        new_miner_resumes = {}
        for uid, miner_resume in miner_resumes.items():
            resume_extractor.add_resume_data(miner_resume)
            extracted_resume = resume_extractor.get_segments()
            new_miner_resumes[uid] = extracted_resume
        return new_miner_resumes

    def process_job_description(self, job_description: str) -> Dict[str, Any]:
        skills_df = self.jd_keys.get_skills_dataframe(job_description=job_description)
        processed_job_description = self.jd_keys.get_formatted_jd(df=skills_df)
        jd_skills = JDSkills(skills_df, job_description)
        universal_skills_weights, preferred_skills_weights = jd_skills.get_skills_weights()
        scoring_data = {
            'jd': processed_job_description,
            'skills': skills_df,
            'universal': universal_skills_weights,
            'preferred': preferred_skills_weights
        }
        return scoring_data

    async def get_job_description(self) -> Dict[str, Any]:
        client = ModuleClient(host="213.173.105.83", port=56709, key=self.key)

        ss58 = self.key.ss58_address
        timestamp = time.time()
        data = f"{ss58}{timestamp}"
        signature = self.key.sign(data.encode())

        try:
            api_response = await client.call(
                "get_prompt",
                "5E5LLGg2jFesFCQsn5N4jCS1Rng4MFxRuBVyvb2zWEETAUbj",
                {"ss58": ss58, "timestamp": timestamp, "signature": signature.hex()},
                timeout=self.call_timeout+120
            )
        except Exception as e:
            print("WARNING: JD API Connection Error: getting record from public hf repo"
                  "(most likely validator has not set weights and is not recognized yet; "
                  f"if issue persists for multiple steps, message reai's discord channel)")
            dataset = datasets.load_dataset("nakamoto-yama/job-descriptions-public")
            api_response = dataset['train'].shuffle(seed=42).select([0])[0]
        print(f"API Response: {api_response}")
        return api_response['Description']

    def vote(self, uids: List[int], weights: List[int]):
        self.client.vote(key=self.key, uids=uids, weights=weights, netuid=self.netuid, use_testnet=self.use_testnet)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="reai validator")
    parser.add_argument("--env", type=str, default=".env", help="config file path")
    parser.add_argument('--ignore-env-file', action='store_true', help='If set, ignore .env file')
    args = parser.parse_args()

    home_dir = os.path.expanduser("~")
    commune_dir = os.path.join(home_dir, ".commune")
    reai_dir = os.path.join(commune_dir, "reai")

    config = ValidatorConfig(env_path=args.env, ignore_config_file=args.ignore_env_file)

    logger.info("Downloading punkt...")
    nltk.download('punkt')
    nltk.download('punkt_tab')
    logger.info("Finished")

    try:
        keypair = classic_load_key(config.get_key_name())
        client = ComxClient(client=CommuneClient(get_node_url(use_testnet=config.get_testnet())))

        validator = Validator(
            key=keypair,
            netuid=config.get_netuid(),
            client=client,
            weight_io=WeightIO(io=IO(), dir_path=reai_dir, file_name="weights.json"),
            interval=config.get_validator_interval(),
            call_timeout=20,
            use_testnet=config.get_testnet()
        )

        validator.validation_loop()
    except ValueError as e:
        print(e)
