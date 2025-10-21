import os
from web3 import Web3
import json
import csv
import sys
import module.TokenPairEventParser as TokenPairEventParser

ETHERSCAN_KEY = os.getenv("ETHERSCAN_KEY")
ALCHEMY_KEY = os.getenv("ALCHEMY_KEY")
API_URL = 'https://api.etherscan.io/v2/api'
LIMIT_NUM = 10000 # 최대 수집 Token 수   

if not ETHERSCAN_KEY or not ALCHEMY_KEY:
    print("[!] Please Input Your API Key via environment variables:")
    print("    export ETHERSCAN_KEY=xxxx")
    print("    export ALCHEMY_KEY=xxxx")
    sys.exit(1)
    
if ALCHEMY_KEY == 'ALCHEMY_KEY' or ETHERSCAN_KEY == 'ETHERSCAN_KEY':
    print('[!] Please Input Your API Key')
    sys.exit(1)

with open(sys.argv[1],'r') as f:
    token_list_json = json.load(f)

start_num = 0
w3 = Web3(Web3.HTTPProvider(f'https://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_KEY}'))

if len(sys.argv) == 3:
    start_num = int(sys.argv[2])
    pair_evt_csv = open(f"pair_evt.csv",'a',newline='',encoding='utf-8')
    token_info_csv = open(f"token_information.csv",'a',newline='',encoding='utf-8')

    pair_writer = csv.writer(pair_evt_csv)
    ti_writer = csv.writer(token_info_csv)
else:
    pair_evt_csv = open(f"pair_evt.csv",'w',newline='',encoding='utf-8')
    token_info_csv = open(f"token_information.csv",'w',newline='',encoding='utf-8')

    pair_writer = csv.writer(pair_evt_csv)
    ti_writer = csv.writer(token_info_csv)

    ti_csv_title = ['token_addr_idx','token_addr','pair_addr','token_create_ts','lp_create_ts','pair_idx']
    ti_writer.writerow(ti_csv_title)

    pair_csv_title = ['token_addr_idx','timestamp','block_number','tx_hash', 'tx_from','tx_to','evt_idx','evt_type','evt_log','token0','token1','reserve0','reserve1','lp_total_supply','evt_removed']
    pair_writer.writerow(pair_csv_title)

un_pair_list = {}
current_idx = 0
cnt = 0

try:
    for token in token_list_json:
        if int(token_list_json[token]) < start_num:
            continue
        
        # etherscan_credits = check_api_usage()
        if cnt == LIMIT_NUM:
            print()
            print(f'Total: {cnt}')
            print(f'To do: {token}({token_list_json[token]})')
            break
        print(f'\rCurrent Num: {token_list_json[token]}',end="",flush=True)
        current_idx = int(token_list_json[token])
        token_addr = Web3.to_checksum_address(token)

        TokenPairEventParser.parser(token_addr,token_list_json[token],ti_writer,pair_writer,w3,ETHERSCAN_KEY)
        
        cnt += 1

finally:
    print()
    print(f'Total: {cnt}')
    print(f'To do: {current_idx}')
    pair_evt_csv.close()
    token_info_csv.close()
