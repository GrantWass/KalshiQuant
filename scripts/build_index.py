"""
scripts/build_index.py — One-shot script to build the FAISS market index.

Run this once before starting the full pipeline to pre-populate the index.
Subsequent refreshes happen automatically every 15 minutes via the pipeline.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --info     # print index info without rebuilding
"""

import argparse
import asyncio
import logging
import sys
import os

# Allow running from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.pool import init_pool, close_pool
from kalshi.auth import token_manager
from kalshi.client import KalshiClient
from embeddings.index import faiss_index
from embeddings.market_index_builder import build_index, load_or_build_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("build_index")


async def main(args: argparse.Namespace) -> None:
    await init_pool()
    await token_manager.start()
    client = KalshiClient()

    try:
        if args.info:
            # Load from disk and print info
            if faiss_index.load():
                print(f"Index loaded: {faiss_index.total_markets} markets")
            else:
                print("No saved index found.")
            return

        if args.force or not faiss_index.load():
            logger.info("Building FAISS index from Kalshi API...")
            await build_index(client)
        else:
            logger.info("Existing index loaded (%d markets). Use --force to rebuild.", faiss_index.total_markets)

        print(f"\nIndex ready: {faiss_index.total_markets} markets indexed.")
        print(f"Saved to: {os.path.abspath('faiss_data/')}")

    finally:
        await client.close()
        await token_manager.stop()
        await close_pool()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the KalshiQuant FAISS market index.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if an index already exists.")
    parser.add_argument("--info", action="store_true", help="Print index info without rebuilding.")
    args = parser.parse_args()
    asyncio.run(main(args))
