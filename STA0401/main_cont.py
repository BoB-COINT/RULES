import os
from web3 import Web3
import json
import csv
import sys
import module.TokenContEventParser as TokenContEventParser

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
    token_evt_csv = open(f"token_evt.csv",'a',newline='',encoding='utf-8')
    token_writer = csv.writer(token_evt_csv)

else:
    token_evt_csv = open(f"token_evt.csv",'w',newline='',encoding='utf-8')
    token_writer = csv.writer(token_evt_csv)

    token_csv_title = ['token_addr_idx','timestamp','block_number','tx_hash', 'tx_from','tx_to','evt_idx','evt_type','evt_log']
    token_writer.writerow(token_csv_title)


un_pair_list = {}
current_idx = 0
cnt = 0

try:
    for token in token_list_json:
        if int(token_list_json[token]) < start_num:
            continue
        
        if cnt == LIMIT_NUM:
            print()
            print(f'Total: {cnt}')
            print(f'To do: {token}({token_list_json[token]})')
            break
        print(f'\rCurrent Num: {token_list_json[token]}',end="",flush=True)
        
        current_idx = int(token_list_json[token])
        token_addr = Web3.to_checksum_address(token)

        TokenContEventParser.parser(token_addr,token_list_json[token],token_writer,w3,ETHERSCAN_KEY)

        cnt += 1

finally:
    print()
    print(f'Total: {cnt}')
    print(f'To do: {current_idx}')
    token_evt_csv.close()
