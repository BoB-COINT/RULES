#!/usr/bin/env python3
"""
debug_swap_parsing.py
---------------------
Token 6, 7의 Swap 파싱이 왜 안되는지 디버깅
"""

import pandas as pd
import json
from decimal import Decimal

def parse_decimal(value):
    if value is None or str(value).strip() == "":
        return None
    try:
        return Decimal(str(value))
    except:
        return None

# Pair events 로드
df = pd.read_csv('pair_evt.csv')

# Token 6, 7의 Swap 이벤트
for token_idx in [6, 7]:
    print(f"\n{'='*60}")
    print(f"Token {token_idx} Analysis")
    print(f"{'='*60}")
    
    swaps = df[(df['token_addr_idx'] == token_idx) & (df['evt_type'] == 'Swap')]
    
    buy_count = 0
    sell_count = 0
    no_match = 0
    
    for idx, row in swaps.iterrows():
        evt_log = eval(row['evt_log'])
        
        # Decimal 변환
        amount0_in = parse_decimal(evt_log.get('amount0In'))
        amount1_in = parse_decimal(evt_log.get('amount1In'))
        amount0_out = parse_decimal(evt_log.get('amount0Out'))
        amount1_out = parse_decimal(evt_log.get('amount1Out'))
        
        # 조건 체크
        case1 = amount0_out and amount0_out > 0 and amount1_in and amount1_in > 0
        case2 = amount0_in and amount0_in > 0 and amount1_out and amount1_out > 0
        case3 = amount1_out and amount1_out > 0 and amount0_in and amount0_in > 0
        case4 = amount1_in and amount1_in > 0 and amount0_out and amount0_out > 0
        
        is_buy = False
        is_sell = False
        matched_case = None
        
        if case1:
            is_buy = True
            matched_case = "Case 1: amount0Out + amount1In (BUY)"
        elif case2:
            is_sell = True
            matched_case = "Case 2: amount0In + amount1Out (SELL)"
        elif case3:
            is_buy = True
            matched_case = "Case 3: amount1Out + amount0In (BUY)"
        elif case4:
            is_sell = True
            matched_case = "Case 4: amount1In + amount0Out (SELL)"
        else:
            no_match += 1
            matched_case = "NO MATCH"
        
        if is_buy:
            buy_count += 1
        elif is_sell:
            sell_count += 1
        
        # 첫 5개만 상세 출력
        if (buy_count + sell_count + no_match) <= 5:
            print(f"\nSwap #{buy_count + sell_count + no_match}:")
            print(f"  amount0In:  {amount0_in}")
            print(f"  amount1In:  {amount1_in}")
            print(f"  amount0Out: {amount0_out}")
            print(f"  amount1Out: {amount1_out}")
            print(f"  → {matched_case}")
            print(f"  → Result: {'BUY' if is_buy else 'SELL' if is_sell else 'UNKNOWN'}")
    
    print(f"\n{'='*60}")
    print(f"Summary for Token {token_idx}:")
    print(f"  Total Swaps: {len(swaps)}")
    print(f"  Buy:  {buy_count}")
    print(f"  Sell: {sell_count}")
    print(f"  No Match: {no_match}")
    print(f"{'='*60}")