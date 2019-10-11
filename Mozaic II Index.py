#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import warnings
import math
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
dataset = pd.ExcelFile("NewDataset.xlsx")


# In[ ]:


# 目标波动率
TARGET_VOLATILITY = 0.042
# 最大仓位
MAX_WEIGHT = 3.0
# 交易手续费
FEE_RATE = 0.0
# 取收益率排名前NN个资产
NN = 9


# In[ ]:


# 读取数据
sp = pd.read_excel(dataset, sheet_name = "S&P500", skiprows = 2)
nq = pd.read_excel(dataset, sheet_name = "NASDAQ", skiprows = 2)
rl = pd.read_excel(dataset, sheet_name = "RUSSELL", skiprows = 2) 
dx = pd.read_excel(dataset, sheet_name = "DAX", skiprows = 2) #EUR
ft = pd.read_excel(dataset, sheet_name = "FTSE", skiprows = 2) #GBP
tx = pd.read_excel(dataset, sheet_name = "TOPIX", skiprows = 2) #JPY
twoyr = pd.read_excel(dataset, sheet_name = "2YR", skiprows = 2)
fiveyr = pd.read_excel(dataset, sheet_name = "5YR", skiprows = 2) 
tenyr = pd.read_excel(dataset, sheet_name = "10YR", skiprows = 2) 
eb = pd.read_excel(dataset, sheet_name = "EuroBond", skiprows = 2) #EUR
gt = pd.read_excel(dataset, sheet_name = "GILT", skiprows = 2) #GBP
jb = pd.read_excel(dataset, sheet_name = "JGB", skiprows = 2) #JPY
ns = pd.read_excel(dataset, sheet_name = "NonSecu", skiprows = 2) 
eurusd = pd.read_excel(dataset, sheet_name = "EURUSD", skiprows = 2)
usdjpy = pd.read_excel(dataset, sheet_name = "USDJPY", skiprows = 2)
gbpusd = pd.read_excel(dataset, sheet_name = "GBPUSD", skiprows = 2) 


# In[ ]:


# 为 non securities 构建时间序列
def ns_time_series(target, loc, name):
    pos = 3
    templst = [1000]
    while target.iloc[pos, 0].year < 1999:
        pos += 1
    date = [target.iloc[pos,0]]
    for i in range(pos + 1, len(target)):
        date.append(target.iloc[i, 0])
        templst.append(templst[-1] * float(target.iloc[i, loc]) / float(target.iloc[i-1, loc]))
    return pd.DataFrame({'Date': date, name: templst})


# In[ ]:


# 调整汇率影响
def adjust_fx(target, rates, cur, offset):
    rates = rates.iloc[offset:]
    ls1 = target.columns.tolist()
    ls2 = rates.columns.tolist()
    ls1[0] = "Date"
    ls2[0] = "Date"
    target.columns = ls1
    rates.columns = ls2
    target['Date'] = pd.to_datetime(target['Date'], utc = True)
    rates['Date'] = pd.to_datetime(rates['Date'], utc = True)
    tempframe = target.merge(rates.iloc[offset:], left_on='Date', right_on='Date')
    for i in range(len(tempframe)):
        if cur == 'EUR' or cur == 'GBP':
            tempframe.iloc[i,1] *= float(tempframe.iloc[i,-1])/float(tempframe.iloc[0,-1])
        else:
            tempframe.iloc[i,1] /= float(tempframe.iloc[i,-1])/float(tempframe.iloc[0,-1])
    return tempframe.drop([tempframe.columns[-1]], axis = 1)


# In[ ]:


# 期货合约到期时间
def expiration_time(ticker, prefix):
    month = letter_to_month(ticker[len(prefix)])
    year = num_to_year(ticker[len(prefix)+1 : len(prefix)+3])
    return year, month


# In[ ]:


# 判断期货合约月份
def letter_to_month(L):
    return{
        'F' : 1,
        'G' : 2,
        'H' : 3,
        'J' : 4,
        'K' : 5,
        'M' : 6,
        'N' : 7,
        'Q' : 8,
        'U' : 9,
        'V' : 10,
        'X' : 11,
        'Z' : 12
    }.get(L)


# In[ ]:


# 判断期货合约年份
def num_to_year(string):
    num = string.strip()
    if len(num) == 2 and (num[0] == '8' or num[0] == '9'):
        return int("19" + num)
    if len(num) == 2 and (num[0] == '0' or num[0] == '1'):
        return int("20" + num)
    if len(num) == 1 and int(num) >= 8:
        return int("201" + num)
    return int("202" + num)


# In[ ]:


# 构建时间序列
def create_timeseries(frame, vol_start, pri_start, num_col, row_offset, ticker, dist_vol_exp):
    rownum = row_offset
    vol_col = vol_start
    pri_col = pri_start
    main_exp_yr, main_exp_mo = expiration_time(frame.iloc[0, vol_col - dist_vol_exp], ticker)
    while np.isnan(frame.iloc[rownum, pri_col]) == True or np.isnan(frame.iloc[rownum, vol_col]) == True or frame.iloc[rownum, 0].year < 1999:
        if frame.iloc[rownum, 0].year == main_exp_yr and frame.iloc[rownum, 0].month == main_exp_mo:
            vol_col += num_col
            pri_col += num_col
            main_exp_yr, main_exp_mo = expiration_time(frame.iloc[0, vol_col - dist_vol_exp], ticker)
        rownum += 1
    vol_vice = 0
    pri_vice = 0
    vol_count = 0
    force_change = 0
    vice_exp_yr = 0
    vice_exp_mo = 0
    vice_determined = 0
    date = [frame.iloc[rownum, 0]]
    idx = [1000.0]
    ret = [0.0]
    for i in range(rownum+1, len(frame)-3):
        date.append(frame.iloc[i,0])
        if vice_determined == 0:
            span = 0
            while np.isnan(frame.iloc[i, pri_col + span * num_col]) == False and np.isnan(frame.iloc[i, vol_col + span * num_col]) == False and np.isnan(frame.iloc[i, vol_col + span * num_col - 1]) == False and ((frame.iloc[i, pri_col + span * num_col] != frame.iloc[i+1, pri_col + span * num_col]) or (frame.iloc[i+2, pri_col + span * num_col] != frame.iloc[i+1, pri_col + span * num_col])) and pri_col + (span + 1) * num_col < len(frame.iloc[i]):
                span += 1
            if span > 1:
                vice_determined = 1
                vol_list = [frame.iloc[i, vol_col + num_col * j] for j in range(1, span)]
                vol_vice = vol_col + (vol_list.index(max(vol_list)) + 1) * num_col
                pri_vice = pri_col + (vol_list.index(max(vol_list)) + 1) * num_col
                vice_exp_yr, vice_exp_mo = expiration_time(frame.iloc[0, vol_vice - dist_vol_exp], ticker)
        if vice_determined == 0:
            ret.append(float(frame.iloc[i, pri_col] / frame.iloc[i - 1, pri_col]) - 1.0)
            idx.append(idx[-1] * (1.0 + ret[-1]))
        else:
            if main_exp_yr == frame.iloc[i, 0].year and main_exp_mo == frame.iloc[i+3, 0].month and vol_count <= 3:
                force_change += 1
                ret.append(float((3 - force_change) / 3) * (frame.iloc[i, pri_col] / frame.iloc[i-1, pri_col]) + float(force_change / 3) * (frame.iloc[i, pri_vice] / frame.iloc[i-1, pri_vice]) - 1.0)
                idx.append(idx[-1] * (1.0 + ret[-1]))
                if force_change == 3:
                    force_change = 0
                    vol_count = 0
                    pri_col = pri_vice
                    vol_col = vol_vice
                    main_exp_yr = vice_exp_yr
                    main_exp_mo = vice_exp_mo
                    vice_determined = 0
                    continue
                else:
                    continue
            if frame.iloc[i,vol_vice] < frame.iloc[i,vol_col]:
                vol_count = 0
                ret.append(float(frame.iloc[i, pri_col] / frame.iloc[i - 1, pri_col]) - 1.0)
                idx.append(idx[-1] * (1.0 + ret[-1]))
            else:
                if vol_count < 3:
                    vol_count += 1
                    ret.append(float(frame.iloc[i, pri_col] / frame.iloc[i - 1, pri_col]) - 1.0)
                    idx.append(idx[-1] * (1.0 + ret[-1]))
                else:
                    vol_count += 1
                    ret.append(float((6 - vol_count) / 3) * (frame.iloc[i, pri_col] / frame.iloc[i-1, pri_col]) + float((vol_count - 3) / 3) * (frame.iloc[i, pri_vice] / frame.iloc[i-1, pri_vice]) - 1.0)
                    idx.append(idx[-1] * (1.0 + ret[-1]))
                    if vol_count == 6:
                        vol_count = 0
                        force_change = 0
                        pri_col = pri_vice
                        vol_col = vol_vice
                        main_exp_yr = vice_exp_yr
                        main_exp_mo = vice_exp_mo
                        vice_determined = 0
    return pd.DataFrame({
        'Date': date,
        'Index': idx,
    })


# In[ ]:


# 构建各产品时间序列
TWO_YR_T_SERIES = create_timeseries(twoyr, 2, 5, 5, 3, 'TU', 1)
FIVE_YR_T_SERIES = create_timeseries(fiveyr, 2, 5, 5, 3, 'FV', 1)
TEN_YR_T_SERIES = create_timeseries(tenyr, 2, 5, 5, 3, 'TY', 1)
SP_SERIES = create_timeseries(sp, 2, 5, 5, 3, 'ES', 1)
NASDAQ_SERIES = create_timeseries(nq, 2, 5, 5, 3, 'NQ', 1)
DAX_SERIES = create_timeseries(dx, 2, 5, 5, 3, 'GX', 1) #EUR
FTSE_SERIES = create_timeseries(ft, 2, 5, 5, 3, 'Z ', 1) #GBP
TOPIX_SERIES = create_timeseries(tx, 2, 5, 5, 3, 'TP', 1) #JPY
EUROBOND_SERIES = create_timeseries(eb, 2, 5, 5, 3, 'RX', 1) #EUR
GILT_SERIES = create_timeseries(gt, 2, 5, 5, 3, 'G ', 1) #GBP
JGB_SERIES = create_timeseries(jb, 2, 5, 5, 3, 'JB', 1) #JPY
RUSSELL_INDEX_SERIES = ns_time_series(rl, 5, "RUSSELL")
BBG_ENERGY_SERIES = ns_time_series(ns, 5, "BCOMEN")
BBG_IMETAL_SERIES = ns_time_series(ns, 10, "BCOMIN")
BBG_PMETAL_SERIES = ns_time_series(ns, 15, "BCOMPR")


# In[ ]:


# 调整汇率影响
DAX_SERIES = adjust_fx(DAX_SERIES, eurusd, 'EUR', 4)
EUROBOND_SERIES = adjust_fx(EUROBOND_SERIES, eurusd, 'EUR', 4)
FTSE_SERIES = adjust_fx(FTSE_SERIES, gbpusd, 'GBP', 4)
GILT_SERIES = adjust_fx(GILT_SERIES, gbpusd, 'GBP', 4)
TOPIX_SERIES = adjust_fx(TOPIX_SERIES, usdjpy, 'JPY', 4)
JGB_SERIES = adjust_fx(JGB_SERIES, usdjpy, 'JPY', 4)


# In[ ]:


# 合并价格时间序列
newdataset = TWO_YR_T_SERIES.merge(FIVE_YR_T_SERIES, left_on='Date', right_on='Date')
newdataset = newdataset.merge(TEN_YR_T_SERIES, left_on='Date', right_on='Date')
newdataset = newdataset.merge(SP_SERIES, left_on='Date', right_on='Date')
newdataset = newdataset.merge(NASDAQ_SERIES, left_on='Date', right_on='Date')
newdataset = newdataset.merge(RUSSELL_INDEX_SERIES, left_on='Date', right_on='Date')
newdataset = newdataset.merge(BBG_ENERGY_SERIES, left_on='Date', right_on='Date')
newdataset = newdataset.merge(BBG_IMETAL_SERIES, left_on='Date', right_on='Date')
newdataset = newdataset.merge(BBG_PMETAL_SERIES, left_on='Date', right_on='Date')
newdataset['Date'] = pd.to_datetime(newdataset['Date'], utc = True)
newdataset = newdataset.merge(DAX_SERIES, left_on='Date', right_on='Date')
newdataset = newdataset.merge(EUROBOND_SERIES, left_on='Date', right_on='Date')
newdataset = newdataset.merge(FTSE_SERIES, left_on='Date', right_on='Date')
newdataset = newdataset.merge(GILT_SERIES, left_on='Date', right_on='Date')
newdataset = newdataset.merge(TOPIX_SERIES, left_on='Date', right_on='Date')
newdataset = newdataset.merge(JGB_SERIES, left_on='Date', right_on='Date')
clnlst = ['Date', 'TREASURY_2YR', 'TREASURY-5YR' , 'TREASURY-10YR', 'S&P500','NASDAQ100','RUSSELL2000', 'BBG_ENERGY', 'BBG_INDUSTRY','BBG_PRECIOUS','DAX','EURO_BOND','FTSE100','GILT', 'TOPIX', 'JAPAN_BOND']
newdataset.columns = clnlst


# In[ ]:


# 构建日收益率序列
daily_return = newdataset.copy()
for i in range(1, len(daily_return.iloc[0])):
    daily_return.iloc[0,i] = 0.0
for i in range(1, len(daily_return)):
    for j in range(1, len(daily_return.iloc[0])):
        daily_return.iloc[i,j] = (newdataset.iloc[i,j] / newdataset.iloc[i-1,j]) - 1.0


# In[ ]:


# 输出收益率指数和收益率序列
newdataset.to_csv('total_return_idx.csv', index = False, header = True)
daily_return.to_csv('daily_return.csv', index = False, header = True)


# In[ ]:


# 计算历史收益波动率
def hist_volatility(returnseries, current):
    lst1 = returnseries.iloc[current-129: current+1].tolist()
    lst2 = returnseries.iloc[current-64: current+1].tolist()
    lst3 = returnseries.iloc[current-21: current+1].tolist()
    r_bar1 = np.mean(lst1)
    r_bar2 = np.mean(lst2)
    r_bar3 = np.mean(lst3)
    return max(math.sqrt(252.0/129.0 * np.sum([(r_bar1 - i)**2 for i in lst1])), math.sqrt(252.0/64.0 * np.sum([(r_bar2 - j)**2 for j in lst2])), math.sqrt(252.0/21.0 * np.sum([(r_bar3 - k)**2 for k in lst3])))


# In[ ]:


# 判断当前日期是否为调仓日
def trading_day(pricepenal, current):
    if pricepenal.iloc[current, 0].month != pricepenal.iloc[current-1, 0].month:
        return True
    else:
        return False


# In[ ]:


# 区间累计收益率
def cumu_return(priceseries, current):
    return (float(priceseries.iloc[current]) / float(priceseries.iloc[current-124])) - 1.0


# In[ ]:


# 收益率排序
def sort_ret(lst, low, high):
    i = low - 1
    if low < high:
        pivot = lst[high][1]
        for j in range(low, high):
            if lst[j][1] >= pivot:
                i += 1
                temp = lst[i]
                lst[i] = lst[j]
                lst[j] = temp
        temp = lst[i+1]
        lst[i+1] = lst[high]
        lst[high] = temp
        sort_ret(lst, low, i)
        sort_ret(lst, i+2, high)


# In[ ]:


# 单个资产最大权重
def security_max_weight(secu_id):
    if int(secu_id) in [4, 5, 6, 7, 8, 9, 10, 12, 14]:
        return 0.15
    elif int(secu_id) in [3, 11, 13]:
        return 0.45
    elif int(secu_id) == 2:
        return 0.75
    elif int(secu_id) == 15:
        return 1.2
    elif int(secu_id) == 1:
        return 2.5
    else:
        print("ERROR")


# In[ ]:


# 输出本次调仓 weight 不为0的合约代码及对应的目标仓位
def ret_signal(pricepenal, retpenal, current, portfolio_ret):
    if len(portfolio_ret) > 130:
        realized_portfolio_vol = hist_volatility(pd.DataFrame(portfolio_ret).iloc[:, 0], len(portfolio_ret) - 1)
    else:
        realized_portfolio_vol = 0.01
    secu_lst = []
    for security in range(1, len(pricepenal.iloc[1])):
        secu_lst.append((security, cumu_return(pricepenal.iloc[:,security], current), hist_volatility(retpenal.iloc[:,security], current)))
    sort_ret(secu_lst, 0, len(secu_lst)-1)
    sum_of_prelim_weight = np.sum([float(1/NN) * (TARGET_VOLATILITY / item[2]) for item in secu_lst])
    result = []
    for i in range(NN):
        if secu_lst[i][1] > 0:
            prelim_weight = float(1 / NN) * TARGET_VOLATILITY / secu_lst[i][2]
            multiplier = min(TARGET_VOLATILITY / realized_portfolio_vol, MAX_WEIGHT / sum_of_prelim_weight, security_max_weight(secu_lst[i][0])/prelim_weight)
            result.append((secu_lst[i][0], prelim_weight * multiplier))
    return result


# In[ ]:


# 执行交易
def execution(net_value_old, asset_old, liability_old, cash_old, portfolio_ori, portfolio_fut, pricepenal, current):
    portfolio_target = list(portfolio_fut)
    portfolio_old = list(portfolio_ori)
    transaction = 0.0
    net_value_new = 0
    asset_new = 0
    liability_new = float(liability_old)
    cash_new = float(cash_old)
    for item in portfolio_old:
        asset_new += item[3] * pricepenal.iloc[current, item[0]]
    asset_new += cash_old
    net_value_new = asset_new - liability_old
    portfolio_new = []
    
    # 计算目标仓位所需数量
    for i in range(len(portfolio_target)):
        portfolio_target[i] = tuple_change_value(portfolio_target, i, 3, net_value_new * portfolio_target[i][2] / pricepenal.iloc[current, portfolio_target[i][0]])

    # 比较新旧投资组合，计算损益、交易手续费
    for i in range(len(portfolio_target)):
        match = 0
        for j in range(len(portfolio_old)):
            if portfolio_old[j][0] == portfolio_target[i][0]:
                match = 1
                portfolio_old[j] = tuple_change_value(portfolio_old, j, 0, 999)
                if portfolio_target[i][3] > portfolio_old[j][3]:
                    portfolio_new.append(portfolio_target[i])
                    temp = (portfolio_target[i][3] - portfolio_old[j][3]) * pricepenal.iloc[current, portfolio_target[i][0]]
                    if cash_new >= temp:
                        cash_new -= temp
                        transaction += temp * FEE_RATE
                    else:
                        asset_new += temp - cash_new
                        liability_new += temp - cash_new
                        cash_new = 0
                        transaction += temp * FEE_RATE
                else:
                    portfolio_new.append(portfolio_target[i])
                    temp = (portfolio_old[j][3] - portfolio_target[i][3]) * pricepenal.iloc[current, portfolio_target[i][0]]
                    cash_new += temp
                    transaction += temp * FEE_RATE
                break
        if match == 0:
            portfolio_new.append(portfolio_target[i])
            temp = portfolio_target[i][3] * pricepenal.iloc[current, portfolio_target[i][0]]
            if cash_new >= temp:
                cash_new -= temp
                transaction += temp * FEE_RATE
            else:
                asset_new += temp - cash_new
                liability_new += temp - cash_new
                cash_new = 0
                transaction += temp * FEE_RATE
    
    # 计算交易手续费造成的净值损失
    for i in range(len(portfolio_old)):
        if portfolio_old[i][0] != 999:
            temp = portfolio_old[i][3] * pricepenal.iloc[current, portfolio_old[i][0]]
            cash_new += temp
            transaction += temp * FEE_RATE
            liability_new += transaction
            net_value_new -= transaction
    
    # 调整剩余现金
    if cash_new > 0 and asset_new > net_value_new:
        temp1 = min(cash_new, liability_new)
        cash_new -= temp1
        liability_new -= temp1
        asset_new -= temp1
    return net_value_new, asset_new, liability_new, cash_new, portfolio_new                


# In[ ]:


def trade(pricepenal, retpenal):
    date = []
    tot_ret_idx = []
    leverage = []
    cash_level = []
    holding = []
    daily_ret = []
    today = 0
    while pricepenal.iloc[today, 0].year != 2000:
        today += 1
    net_value = 1000.0
    liability = 0.0
    asset = 1000.0
    cash = 1000.0
    portfolio = []  # (标的，目标仓位，实际仓位，数量)
    stop_loss = 1
    stop_loss_day_count = 0
    stop_loss_multiplier = 1
    stop_loss_rev = 0
    
    
    while pricepenal.iloc[today, 0].year != 2019 or pricepenal.iloc[today, 0].month != 9:
        
        date.append(pricepenal.iloc[today, 0])
        
        # 止损系统
        if stop_loss == 0:
            if stop_loss_day_count == 1:
                stop_loss_multiplier = 0.67
            elif stop_loss_day_count == 2:
                stop_loss_multiplier = 0.33
            elif stop_loss_day_count >= 3 and stop_loss_day_count <= 5:
                stop_loss_multiplier = 0
            else:
                stop_loss = 1
                stop_loss_rev = 1
                stop_loss_day_count = 0
                stop_loss_multiplier = 1
                
        # 调仓日动作
        if trading_day(pricepenal, today) == True:
            portfolio_target = []
            temp = ret_signal(pricepenal, retpenal, today, daily_ret)
            for item in temp:
                portfolio_target.append((item[0], item[1], item[1] * stop_loss_multiplier, 0.0))
            (net_value, asset, liability, cash, portfolio) = execution(net_value, asset, liability, cash, portfolio, portfolio_target, pricepenal, today)
            tot_ret_idx.append(net_value)
            if len(tot_ret_idx) > 1:
                daily_ret.append((tot_ret_idx[-1]/tot_ret_idx[-2])-1.0)
            else:
                daily_ret.append(0)
            holding.append(portfolio)
            cash_level.append(cash)
            leverage.append(asset / net_value)
            stop_loss_rev = 0

        # 非调仓日计算损益及止损调仓    
        else:
            if (stop_loss == 0 and stop_loss_day_count <= 4) or stop_loss_rev == 1:
                portfolio_tgt = list(portfolio)
                for i in range(len(portfolio_tgt)):
                    portfolio_tgt[i] = tuple_change_value(portfolio_tgt, i, 2, portfolio_tgt[i][1] * stop_loss_multiplier)
                (net_value, asset, liability, cash, portfolio) = execution(net_value, asset, liability, cash, portfolio, portfolio_tgt, pricepenal, today)
                tot_ret_idx.append(net_value)
                daily_ret.append((tot_ret_idx[-1]/tot_ret_idx[-2])-1.0)
                holding.append(portfolio)
                cash_level.append(cash)
                leverage.append(asset / net_value)
                stop_loss_rev = 0
            else:
                asset = 0.0
                for i in range(len(portfolio)):
                    asset += portfolio[i][3] * pricepenal.iloc[today, portfolio[i][0]]
                asset += cash
                net_value = asset - liability
                tot_ret_idx.append(net_value)
                daily_ret.append((tot_ret_idx[-1]/tot_ret_idx[-2])-1.0)
                holding.append(portfolio)
                cash_level.append(cash)
                leverage.append(asset / net_value)
        
        # 判断是否触发止损
        if stop_loss == 0:
            stop_loss_day_count += 1
        elif len(tot_ret_idx) > 4:
            if tot_ret_idx[-1] / max(tot_ret_idx[-5:]) < 0.97:
                stop_loss = 0
                stop_loss_day_count += 1
        
        today += 1

    ret_holding = []
    secu_name = pricepenal.columns.tolist()
    for i in range(len(holding)):
        ret_holding.append([(secu_name[item[0]], item[3]) for item in holding[i]])
    
    
    high_water_mark = 0
    mdd = 0
    mdd_lst = []
    for i in range(len(tot_ret_idx)):
        if tot_ret_idx[i] >= high_water_mark:
            high_water_mark = tot_ret_idx[i]
        if 1.0 - (tot_ret_idx[i] / high_water_mark) > mdd:
            mdd = 1.0 - (tot_ret_idx[i] / high_water_mark)
        mdd_lst.append(mdd)
    
            
    return pd.DataFrame({
        'Date': date,
        'Total Return Index': tot_ret_idx,
        'Daily Return': daily_ret,
        'Max Drawdown': mdd_lst,
        'Leverage': leverage,
        'Cash': cash_level,
        'Holdings_EOD': ret_holding
    })
    
    


# In[ ]:


def tuple_change_value(portfolio_ori, idx, idx2, new_value):
    lst = list(portfolio_ori[idx])
    lst[idx2] = new_value
    return tuple(lst)


# In[ ]:


resulting_index = trade(newdataset, daily_return)
resulting_index.to_csv('mozaic2_index.csv', index = False, header = True)

