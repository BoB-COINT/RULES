from web3 import Web3
import requests
import json
from datetime import datetime, timezone
import time

SIG_LIB = {
    '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef':'Transfer',
    '0x8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925':'Approval'
}

TOKEN_ROUTER = {
    '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D': 'UniswapV2',
    '0xE592427A0AEce92De3Edee1F18E0157C05861564': 'UniSwapV3',
    '0x66a9893cC07D91D95644AEDD05D03f95e1dBA8Af': 'UniSwapV4(Universal)',
    '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F': 'SusiswapV2',

}

LAST_LOGIDX = 0
PAGE_10_OVER = False
EVT_MAXIMUM = 10000

API_URL = 'https://api.etherscan.io/v2/api'
ERC20_ABI = json.loads('[{"constant":true,"inputs":[],"name":"name","outputs":[{"name":"","type":"string"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":false,"inputs":[{"name":"_spender","type":"address"},{"name":"_value","type":"uint256"}],"name":"approve","outputs":[{"name":"","type":"bool"}],"payable":false,"stateMutability":"nonpayable","type":"function"},{"constant":true,"inputs":[],"name":"totalSupply","outputs":[{"name":"","type":"uint256"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":false,"inputs":[{"name":"_from","type":"address"},{"name":"_to","type":"address"},{"name":"_value","type":"uint256"}],"name":"transferFrom","outputs":[{"name":"","type":"bool"}],"payable":false,"stateMutability":"nonpayable","type":"function"},{"constant":true,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":true,"inputs":[{"name":"_owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":true,"inputs":[],"name":"symbol","outputs":[{"name":"","type":"string"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":false,"inputs":[{"name":"_to","type":"address"},{"name":"_value","type":"uint256"}],"name":"transfer","outputs":[{"name":"","type":"bool"}],"payable":false,"stateMutability":"nonpayable","type":"function"},{"constant":true,"inputs":[{"name":"_owner","type":"address"},{"name":"_spender","type":"address"}],"name":"allowance","outputs":[{"name":"","type":"uint256"}],"payable":false,"stateMutability":"view","type":"function"},{"anonymous":false,"inputs":[{"indexed":true,"name":"_from","type":"address"},{"indexed":true,"name":"_to","type":"address"},{"indexed":false,"name":"_value","type":"uint256"}],"name":"Transfer","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"name":"_owner","type":"address"},{"indexed":true,"name":"_spender","type":"address"},{"indexed":false,"name":"_value","type":"uint256"}],"name":"Approval","type":"event"}]')


def get_decimals(w3,token_addr):
    checksum_token = Web3.to_checksum_address(token_addr)
    try:
        erc20 = w3.eth.contract(address=checksum_token, abi=ERC20_ABI)
        decimal = erc20.functions.decimals().call()
    except:
        decimal = 0
    return decimal
def timeformat(timestamp):
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return dt

def get_fromtobytx(w3,txhash):
    tx_info = {}
    data = w3.eth.get_transaction(txhash)
    if data['from'] == None:
        tx_info['from'] = 'None'
    else:
        tx_info['from'] = data['from']

    if data['to'] == None:
        tx_info['to'] = 'None'
    else:
        tx_info['to'] = data['to']

    return tx_info

def get_tokencreate(token_addr,etherscan_key):
    param = {
        "chainid":1,
        "module": "contract",
        "action": "getcontractcreation",
        "contractaddresses": token_addr,
        "apikey": etherscan_key
    }
    
    res = requests.get(API_URL,params=param)
    result = res.json()['result'][0]
    create_time = int(result['timestamp'])
    blocknum = result['blockNumber']

    return create_time,blocknum

def get_block_after_days(w3, start_block,start_time,days):
    target_time = start_time + (days * 24 * 60 * 60)
    current_block = w3.eth.block_number

    # 이진탐색으로 target_time에 가장 가까운 블록 찾기
    low = int(start_block)
    high = int(current_block)

    while low < high:
        mid = (low + high) // 2
        block_time = w3.eth.get_block(mid)['timestamp']
        if block_time < target_time:
            low = mid + 1
        else:
            high = mid
    return low

def get_transfer(log,token,w3):
    json_list = {}
    from_to = ['from','to']
    evt_type = 'Transfer'

    if log['logIndex'] == '0x':
        logindex = 0
    else:
        logindex = int(log['logIndex'],16)
    
    tx_hash = log['transactionHash']
    data  = log['data']
    timestamp = timeformat(int(log['timeStamp'],16))
    blocknum = int(log['blockNumber'],16)
    tx_info = get_fromtobytx(w3,tx_hash)
    topics = log['topics']

    for i in range(2):
        addr = '0x' + topics[i+1][-40:]
        json_list[f'{from_to[i]}'] = addr
    
    if len(topics) == 4:
        amount_store = int(topics[3],16)
    else:
        amount_store = int(data,16)
    
    json_list['value'] = amount_store / 10**token['decimals']

    csvrow = [token['idx'],timestamp,blocknum,tx_hash,tx_info['from'],tx_info['to'],logindex,evt_type,json_list]
    return csvrow

def get_Approval(log,token,w3):
    json_list = {}
    evt_type = 'Approval'
    if log['logIndex'] == '0x':
        logindex = 0
    else:
        logindex = int(log['logIndex'],16)
    tx_hash = log['transactionHash']
    topics = log['topics']

    timestamp = timeformat(int(log['timeStamp'],16))
    blocknum = int(log['blockNumber'],16)
    tx_info = get_fromtobytx(w3,tx_hash)
    owner_spender = ['owner','spender']
    
    for i in range(2):
        addr = '0x' + topics[i+1][-40:]
        json_list[f'{owner_spender[i]}'] = addr

    if len(topics) == 4:
        data = int(topics[3],16)
    else:
        data  = int(log['data'][2:],16)

    spender = Web3.to_checksum_address(json_list['spender'])

    json_list[f'value'] = data/10**token['decimals']

    if spender in TOKEN_ROUTER:
        json_list['Known_router'] = 'True'
    else:
        json_list['Known_router'] = 'False'

    csvrow = [token['idx'],timestamp,blocknum,tx_hash,tx_info['from'],tx_info['to'],logindex,evt_type,json_list]
    return csvrow

def get_evetlog(token,writer,w3,etherscan_key,from_block,to_block):
    global LAST_LOGIDX, PAGE_10_OVER

    current_page = 1
    evt_cnt = 0

    while True:
        param = {
            "chainid":1,
            "module": "logs",
            "action": "getLogs",
            "fromBlock": from_block,
            "toBlock": to_block,
            "page":{current_page},
            "offset":1000,
            "address": token['token_addr'],
            "apikey": etherscan_key
        }
        time.sleep(0.1)
        res = requests.get(API_URL,params=param)
        csvtemp = []
        response = res.json()
        
        while 'status' not in response:
            print(response)
            time.sleep(10)
            res = requests.get(API_URL,params=param)
            response = res.json()
        
        status = int(response['status'])
        message = response['message']

        while status == 0 and message != 'No records found':
            print(response)
            time.sleep(10)
            res = requests.get(API_URL,params=param)
            response = res.json()

            if 'status' not in response:
                continue

            status = int(response['status'])
            message = response['message']

        result = response['result']
        
        if len(result) == 0:
            break

        for log in result:
            sig = log['topics'][0]
            
            if log['logIndex'] == '0x':
                logindex = 0
            else:
                logindex = int(log['logIndex'],16)
            
            blocknum = int(log['blockNumber'],16)

            if sig in SIG_LIB:
                if PAGE_10_OVER:
                    if blocknum == from_block and logindex <= LAST_LOGIDX:
                        continue
                if SIG_LIB[sig] == 'Transfer':
                    if len(log['topics']) == 1:
                        continue
                    temp = get_transfer(log,token,w3)
                    csvtemp.append(temp)
                    evt_cnt += 1
                elif SIG_LIB[sig] == 'Approval':
                    if len(log['topics']) == 1:
                        continue
                    temp = get_Approval(log,token,w3)
                    csvtemp.append(temp)
                    evt_cnt += 1
            if evt_cnt == 10000:
                break
        writer.writerows(csvtemp)
        
        if len(result) < 1000 or evt_cnt == 10000:
            break
        
        if current_page == 10:
            PAGE_10_OVER = True
            last_blocknum = csvtemp[-1][2]
            LAST_LOGIDX = csvtemp[-1][6]
            current_page = 1
            from_block = last_blocknum
            continue

        current_page += 1

def parser(token_addr,token_idx,token_writer,w3,etherscan_key):
    token = {}
    #토큰 주소
    token['token_addr'] = token_addr
    token['decimals'] = get_decimals(w3,token['token_addr'])
    token['idx'] = token_idx
    createtime, blocknum = get_tokencreate(token_addr,etherscan_key)
    to_block = get_block_after_days(w3,blocknum,createtime,14)
    # 토큰 컨트랙트의 Eventlog 수집(Transfer, Approval)
    get_evetlog(token,token_writer,w3,etherscan_key,blocknum,to_block)

    