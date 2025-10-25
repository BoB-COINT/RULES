from web3 import Web3
import requests
import json
from datetime import datetime, timezone
import time

SIG_LIB = {
    '0x4c209b5fc8ad50758f13e2e1088ba56a560dff690a1c6fef26394f4c03821c4f':'Mint',
    '0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822':'Swap',
    '0xdccd412f0b1252819cb1fd330b93224ca42612892bb3f4f789976e6d81936496':'Burn',
    '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef':'Transfer',
    '0x1c411e9a96e071241c2f21f7726b17ae89e3cab4c78be50e062b03a9fffbbad1':'Sync'
}

API_URL = 'https://api.etherscan.io/v2/api'
ERC20_ABI = json.loads('[{"constant":true,"inputs":[],"name":"name","outputs":[{"name":"","type":"string"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":false,"inputs":[{"name":"_spender","type":"address"},{"name":"_value","type":"uint256"}],"name":"approve","outputs":[{"name":"","type":"bool"}],"payable":false,"stateMutability":"nonpayable","type":"function"},{"constant":true,"inputs":[],"name":"totalSupply","outputs":[{"name":"","type":"uint256"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":false,"inputs":[{"name":"_from","type":"address"},{"name":"_to","type":"address"},{"name":"_value","type":"uint256"}],"name":"transferFrom","outputs":[{"name":"","type":"bool"}],"payable":false,"stateMutability":"nonpayable","type":"function"},{"constant":true,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":true,"inputs":[{"name":"_owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":true,"inputs":[],"name":"symbol","outputs":[{"name":"","type":"string"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":false,"inputs":[{"name":"_to","type":"address"},{"name":"_value","type":"uint256"}],"name":"transfer","outputs":[{"name":"","type":"bool"}],"payable":false,"stateMutability":"nonpayable","type":"function"},{"constant":true,"inputs":[{"name":"_owner","type":"address"},{"name":"_spender","type":"address"}],"name":"allowance","outputs":[{"name":"","type":"uint256"}],"payable":false,"stateMutability":"view","type":"function"},{"anonymous":false,"inputs":[{"indexed":true,"name":"_from","type":"address"},{"indexed":true,"name":"_to","type":"address"},{"indexed":false,"name":"_value","type":"uint256"}],"name":"Transfer","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"name":"_owner","type":"address"},{"indexed":true,"name":"_spender","type":"address"},{"indexed":false,"name":"_value","type":"uint256"}],"name":"Approval","type":"event"}]')
PAGE_10_OVER = False
LAST_LOGIDX = 0
EVT_MAXIMUM = 10000


FACTORY_LIST = {
    '0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f': 'UniswapV2',
    '0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac': 'SusiswapV2',
    '0x1097053Fd2ea711dad45caCcc45EfF7548fCB362': 'PancakeV2',
    '0x115934131916C8b277DD010Ee02de363c09d037c': 'ShibaSwapV1',
    '0x9DEB29c9A4c7A88a3C0257393b7f3335338D9A9D': 'DeFi Swap V2',
    '0x43eC799eAdd63848443E2347C49f5f52e8Fe0F6f': 'Fraxwap',
    '0x75e48C954594d64ef9613AeEF97Ad85370F13807': 'SakeSwap',
    '0x69bd16aE6F507bd3Fc9eCC984d50b04F029EF677': 'Whiteswap'
}

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
    create_time = timeformat(int(result['timestamp']))

    return create_time

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

def get_swapevent(log,token_pair,reserve,lp_totalsupply,w3):
    json_list = {}
    in_out = ['In','Out']

    evt_type = 'Swap'
    if log['logIndex'] == '0x':
        logindex = 0
    else:
        logindex = int(log['logIndex'],16)
    tx_hash = log['transactionHash']
    data  = log['data'][2:]
    timestamp = timeformat(int(log['timeStamp'],16))
    blocknum = int(log['blockNumber'],16)

    tx_info = get_fromtobytx(w3,tx_hash)

    for i in range(4):
        amount_store = int('0x' + data[i*64:i*64+64],16)
        decimals = int(token_pair[f'token{int(i%2)}'][1])
        json_list[f'amount{int(i%2)}{in_out[int(i/2)]}'] = amount_store/10**decimals
            
    csvrow = [token_pair['pool_idx'],timestamp,blocknum,tx_hash,tx_info['from'],tx_info['to'],logindex,evt_type,json_list,token_pair['token0'][0],token_pair['token1'][0],reserve[0],reserve[1],lp_totalsupply,'False']

    return csvrow,lp_totalsupply

def get_burn(log,token_pair,reserve,lp_totalsupply,w3):
    json_list = {}
    sender_to = ['sender','to']
    evt_type = 'Burn'
    if log['logIndex'] == '0x':
        logindex = 0
    else:
        logindex = int(log['logIndex'],16)
    tx_hash = log['transactionHash']
    data  = log['data'][2:]
    timestamp = timeformat(int(log['timeStamp'],16))
    blocknum = int(log['blockNumber'],16)
    tx_info = get_fromtobytx(w3,tx_hash)
    topics = log['topics']

    for i in range(len(topics)-1):
        addr = '0x' + topics[i+1][-40:]
        json_list[f'{sender_to[i]}'] = addr

    for i in range(2):
        amount_store = int('0x' + data[i*64:i*64+64],16)
        decimals = int(token_pair[f'token{int(i%2)}'][1])
        json_list[f'amount{i}'] = amount_store/10**decimals
   
    csvrow = [token_pair['pool_idx'],timestamp,blocknum,tx_hash,tx_info['from'],tx_info['to'],logindex,evt_type,json_list,token_pair['token0'][0],token_pair['token1'][0],reserve[0],reserve[1],lp_totalsupply,'False']
    return csvrow,lp_totalsupply

def get_mint(log,token_pair,reserve,lp_totalsupply,w3):
    json_list = {}
    evt_type = 'Mint'
    if log['logIndex'] == '0x':
        logindex = 0
    else:
        logindex = int(log['logIndex'],16)
    tx_hash = log['transactionHash']
    data  = log['data'][2:]
    timestamp = timeformat(int(log['timeStamp'],16))
    blocknum = int(log['blockNumber'],16)
    tx_info = get_fromtobytx(w3,tx_hash)
    topics = log['topics']
    
    for i in range(len(topics)-1):
        addr = '0x' + topics[i+1][-40:]
        json_list[f'argv{i+1}'] = addr

    for i in range(2):
        amount_store = int('0x' + data[i*64:i*64+64],16)
        decimals = int(token_pair[f'token{int(i%2)}'][1])
        json_list[f'amount{i}'] = amount_store/10**decimals
    
    csvrow = [token_pair['pool_idx'],timestamp,blocknum,tx_hash,tx_info['from'],tx_info['to'],logindex,evt_type,json_list,token_pair['token0'][0],token_pair['token1'][0],reserve[0],reserve[1],lp_totalsupply,'False']
    return csvrow,lp_totalsupply

def get_transfer(log,token_pair,reserve,lp_totalsupply,w3):
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

    amount_store = int(data,16)
    value = amount_store / 10**18

    for i in range(len(topics)-1):
        addr = '0x' + topics[i+1][-40:]
        json_list[f'{from_to[i]}'] = addr
        # LP 토큰 생성시 LP TotalSupply 증가
        if int(addr,16) == 0 and i == 0:
            lp_totalsupply += value
        # LP 토큰 소각시 LP TotalSupply 차감
        elif int(addr,16) == 0 and i == 1:
            lp_totalsupply -= value
    
    json_list['value'] = value

    csvrow = [token_pair['pool_idx'],timestamp,blocknum,tx_hash,tx_info['from'],tx_info['to'],logindex,evt_type,json_list,token_pair['token0'][0],token_pair['token1'][0],reserve[0],reserve[1],lp_totalsupply,'False']
    return csvrow,lp_totalsupply

def get_sync(log,token_pair,reserve_list,lp_totalsupply,w3):
    json_list = {}
    evt_type = 'Sync'
    if log['logIndex'] == '0x':
        logindex = 0
    else:
        logindex = int(log['logIndex'],16)
    tx_hash = log['transactionHash']
    data  = log['data'][2:]
    timestamp = timeformat(int(log['timeStamp'],16))
    blocknum = int(log['blockNumber'],16)
    tx_info = get_fromtobytx(w3,tx_hash)

    for i in range(2):
        reserve = int('0x' + data[i*64:i*64+64],16)
        decimals = int(token_pair[f'token{int(i%2)}'][1])
        json_list[f'reserve{i}'] = reserve/10**decimals
        reserve_list[i] = reserve/10**decimals

    csvrow = [token_pair['pool_idx'],timestamp,blocknum,tx_hash,tx_info['from'],tx_info['to'],logindex,evt_type,json_list,token_pair['token0'][0],token_pair['token1'][0],reserve_list[0],reserve_list[1],lp_totalsupply,'False']
    return csvrow,lp_totalsupply

def get_evetlog(token_pair,lp_totalsupply,writer,w3,etherscan_key,start_block,end_block):
    global PAGE_10_OVER
    global LAST_LOGIDX

    checksum_pair = Web3.to_checksum_address(token_pair['token_pair'])
    current_page = 1
    evt_cnt = 0

    while True:
        param = {
            "chainid":1,
            "module": "logs",
            "action": "getLogs",
            "fromBlock": start_block,
            "toBlock": end_block,
            "page":{current_page},
            "offset":1000,
            "address": checksum_pair,
            "apikey": etherscan_key
        }

        time.sleep(0.1)
        res = requests.get(API_URL,params=param)
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

        csvtemp = []
        reserve = [0,0]

        if len(result) == 0:
            break

        for log in result:
            sig = log['topics'][0]
            
            if log['logIndex'] == '0x':
                logindex = 0
            else:
                logindex = int(log['logIndex'],16)
            
            blocknum = int(log['blockNumber'],16)

            if PAGE_10_OVER:
                if blocknum == start_block and logindex <= LAST_LOGIDX:
                    continue
            if sig in SIG_LIB:
                if SIG_LIB[sig] == 'Swap':
                    temp,lp_totalsupply = get_swapevent(log,token_pair,reserve,lp_totalsupply,w3)
                    csvtemp.append(temp)
                    evt_cnt += 1
                elif SIG_LIB[sig] == 'Burn':
                    temp,lp_totalsupply = get_burn(log,token_pair,reserve,lp_totalsupply,w3)
                    csvtemp.append(temp)
                    evt_cnt += 1
                elif SIG_LIB[sig] == 'Mint':
                    temp,lp_totalsupply = get_mint(log,token_pair,reserve,lp_totalsupply,w3)
                    csvtemp.append(temp)
                    evt_cnt += 1
                elif SIG_LIB[sig] == 'Transfer' and ((int(log['topics'][1],16) == 0 and int(log['topics'][2],16) != 0) or (int(log['topics'][1],16) != 0 and int(log['topics'][2],16) == 0)):
                    temp,lp_totalsupply = get_transfer(log,token_pair,reserve,lp_totalsupply,w3)
                    csvtemp.append(temp)
                    evt_cnt += 1
                elif SIG_LIB[sig] == 'Sync':
                    temp,lp_totalsupply = get_sync(log,token_pair,reserve,lp_totalsupply,w3)
                    csvtemp.append(temp)
                    evt_cnt += 1
            if evt_cnt == EVT_MAXIMUM:
                break
        writer.writerows(csvtemp)
        
        if len(result) < 1000 or evt_cnt == EVT_MAXIMUM:
            break

        if current_page == 10:
            PAGE_10_OVER = True
            last_blocknum = csvtemp[-1][2]
            LAST_LOGIDX = csvtemp[-1][6]
            current_page = 1
            start_block = last_blocknum
            continue
        current_page += 1
        
def get_pair(token_addr,token_idx,w3,etherscan_key):
    json_list = {}
    token_pool = {
        'token0':[],
        'token1':[]
    }
    evt_type = 'PairCreated'
    token_addr_enc = '0x' + '0' * 24 + token_addr[2:]

    for factory in FACTORY_LIST:
        for i in range(2):
            param = {
                "chainid":1,
                "module": "logs",
                "action": "getLogs",
                "fromBlock": "0",
                "toBlock": "latest",
                "address": factory,
                "topic0": "0x0d3648bd0f6ba80134a33ba9275ac585d9d315f0ad8355cddefde31afa28d0e9",
                f"topic{i+1}": token_addr_enc,
                "apikey": etherscan_key
            }
            res = requests.get(API_URL,params=param)
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
                
            if len(result) !=0:
                break
            
        if len(result) !=0:
            token_pool['pair_type'] = FACTORY_LIST[factory]
            break
    
    token_createtime = get_tokencreate(token_addr,etherscan_key)

    if len(result) == 0:
        token_pool['pair_type'] = None
        tokeninforow = [token_idx, token_addr, 'None', token_createtime,'None','None']
        return token_pool,None,tokeninforow,None

    if result[0]['logIndex'] == '0x':
        logindex = 0
    else:
        logindex = int(result[0]['logIndex'],16)
    tx_hash = result[0]['transactionHash']
    topics = result[0]['topics']
    data  = result[0]['data']

    token_pool['pair_created_ts'] = int(result[0]['timeStamp'],16)
    
    timestamp = timeformat(token_pool['pair_created_ts'])
    blocknum = int(result[0]['blockNumber'],16)
    token_pool['pair_created_blocknum'] = blocknum

    for i in range(len(topics)-1):
        addr = Web3.to_checksum_address('0x' + topics[i+1][-40:])
        decimals = get_decimals(w3,addr)
        temp = [addr,decimals]
        json_list[f'token{i}'] = addr
        token_pool[f'token{i}'] += temp
        if addr == token_addr:
            pair_idx = i
    
    total_data = data[2:]
    addr = '0x' + total_data[24:64]
    token_pool['token_pair'] = addr
    json_list['pairaddr'] = addr
    tx_info = get_fromtobytx(w3,tx_hash)
    lp_totalsupply = 0
    token_pool['pool_idx'] = token_idx

    tokeninforow = [token_pool['pool_idx'], token_addr, token_pool['token_pair'], token_createtime,timestamp,pair_idx]
    csvrow = [token_pool['pool_idx'],timestamp,blocknum,tx_hash,tx_info['from'],tx_info['to'],logindex,evt_type,json_list,token_pool['token0'][0],token_pool['token1'][0],0,0,lp_totalsupply,'False']

    return token_pool,csvrow,tokeninforow,lp_totalsupply

def parser(token_addr,token_idx,ti_writer,event_writer,w3,etherscan_key):
   
    # 특정 토큰의 Pair 주소 및 Pool 정보 획득
    tp,pair_temp,ti_temp,lp_totalsupply = get_pair(token_addr,token_idx,w3,etherscan_key)
    if tp['pair_type'] == None:
        ti_writer.writerow(ti_temp)
        return

    ti_writer.writerow(ti_temp)
    event_writer.writerow(pair_temp)

    to_block = get_block_after_days(w3,tp['pair_created_blocknum'],tp['pair_created_ts'],14)

    # Pair 주소로 부터 발생된 Eventlog 수집(Transfer, Mint, Burn, Sync, Swap)
    get_evetlog(tp,lp_totalsupply,event_writer,w3,etherscan_key,tp['pair_created_blocknum'],to_block)
    
    return

    
    # print('='*60)
    # print(f'Pair: {tp['token_pair']}')
    # print(f'- Token0: {tp['token0'][0]}')
    # print(f'- Token1: {tp['token1'][0]}')
    # print(f'- Events: Mint={total_event["Mint"]}, Transfer={total_event['Transfer']}, Swap={total_event['Swap']}, Sync={total_event['Sync']}, Burn={total_event['Burn']}')
    # print('='*60)