
import argparse
from urllib.parse import urlparse
import uvicorn
from keylimiter import TokenBucketLimiter
from communex.module.server import ModuleServer
from communex.compat.key import classic_load_key
from config.miner import MinerConfig
from miner.nltk_miner import NltkMiner

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="yama miner")
    parser.add_argument("--env", type=str, default=".env", help="config file path")
    parser.add_argument("--miner", type=str, default="nltk", help="miner type")
    parser.add_argument('--ignore-env-file', action='store_true', help='If set, ignore .env file')
    args = parser.parse_args()

    config = MinerConfig(env_path=args.env, ignore_config_file=False)

    try:
        keypair = classic_load_key(config.get_key_name())
        bucket = TokenBucketLimiter(1000, 1 / 100)

        if args.miner == "nltk":
            miner = NltkMiner()
        elif args.miner == "t5":
            from miner.t5_miner import T5Miner
            miner = T5Miner()
        else:
            miner = NltkMiner()
            print("Unsupported miner, defaulting to NltkMiner")

        server = ModuleServer(
             miner,
             keypair,
             limiter=bucket,
             subnets_whitelist=[config.get_netuid()],
             use_testnet=config.get_testnet()
        )

        parsed_url = urlparse(config.get_miner_url())

        app = server.get_fastapi_app()
        uvicorn.run(app, host=parsed_url.hostname, port=parsed_url.port)
    except ValueError as e:
        print(e)
