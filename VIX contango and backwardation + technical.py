"""
vix contango / backwardation
MACD
market making
"""
import datetime
import numpy as np
import pandas as pd
import talib
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import CustomFactor, Latest
from quantopian.pipeline.data.quandl import cboe_vix, cboe_vxv


class getindex(CustomFactor):
    window_length = 1
    def compute(self, today, assets, out, vix):
        out[:] = vix[-1]

def initialize(context):
    context.asset = sid(8554)
    context.count = 0
    context.burst_pri = 0.1
    context.time = 0
    context.timelap = 0
    context.volthreshold = 800000
    context.position = 0.5
    context.techsignal = 1.0
    pipe = Pipeline()
    attach_pipeline(pipe, 'my_pipeline')
    pipe.add(getindex(inputs=[cboe_vix.vix_close]), 'VixClose')
    pipe.add(getindex(inputs=[cboe_vxv.close]), 'VxvClose')
    schedule_function(func=start, date_rule=date_rules.every_day(), time_rule=time_rules.market_open())
    schedule_function(func=mood, date_rule=date_rules.every_day(), time_rule=time_rules.market_close())
    schedule_function(func=techind, date_rule=date_rules.every_day(), time_rule=time_rules.market_close())
    schedule_function(func=end, date_rule=date_rules.every_day(), time_rule=time_rules.market_close())

def start(context, data):
    context.count = 0
    context.time = 0
    
def end(context, data):
    context.count = 0
    context.time = 0
    context.timelap = 0
    print context.portfolio.capital_used
# 1. If vix is in backwardation, market is likely to be bullish
# 2. If vix > real volatility, market is also likely to be bullish

def mood(context,data):
    context.output = pipeline_output('my_pipeline')
    context.vix = context.output["VixClose"].iloc[0]
    context.vxv = context.output["VxvClose"].iloc[0]
    ind = context.vxv / context.vix
    pri_hist = data.history(context.asset, 'price', 30, '1d')
    real_vol = np.std(pri_hist)
    diff = real_vol * 10 - context.vix
    if ind < 1 and diff < 0:
        context.position = 0.65
    elif ind < 1 or diff < 0:
        context.position = 0.55
    elif diff > 30:
        context.position = 0.40
    elif diff > 20:
        context.position = 0.45
    else:
        context.position = 0.5
    
def techind(context, data):
    price_history = data.history(context.asset, 'price', 1500, '1d')
    acd_raw, signal, macd_hist = talib.MACD(price_history, 
                                             fastperiod=12, 
                                             slowperiod=26, 
                                             signalperiod=9)
    if macd_hist[-1] > 0:
        context.techsignal = 1
    else:
        context.techsignal = 0.8
    

def handle_data(context, data):
    if context.count > 5:
        bear = False
        bull = False
        targetpos = 0
        context.time += 1
        context.price = data.history(context.asset, 'price', 6, '1m')
        vol_lis = data.history(context.asset, 'volume', 2, '1m')
        context.vol = 0.7 * vol_lis[0] + 0.3 * vol_lis[1]
        #adjust vol
        #adjust price
        if context.price[-1] > context.price[-2]:
            orderprice = context.price[-1] + 0.04
        else:
            orderprice = context.price[-1] - 0.04
        #determine if there is a price breakout
        if context.timelap >= 2 and ((context.price[-1] - max(context.price[0:5]) > context.burst_pri) or ((context.price[-1] > max(context.price[0:4]) and context.price[-1] > context.price[-2]))):
            bull = True
            bear = False
        elif context.timelap >= 2 and (min(context.price[0:5]) - context.price[-1]) > context.burst_pri or (((min(context.price[0:4]) - context.price[-1]) > context.burst_pri) and (context.price[-1] < context.price[-2])):
            bear = True
            bull = False
        else:
            bull = False
            bear = False
        #balance portfolio by placing small orders 
        if context.portfolio.capital_used < 0:
            if context.portfolio.positions_value / context.portfolio.portfolio_value < (context.position - 0.02):
                order(context.asset, 15, context.price[-1] + 0.01)
                order(context.asset, 10, context.price[-1] + 0.02)
                order(context.asset, 5, context.price[-1] + 0.03)
            elif context.portfolio.positions_value / context.portfolio.portfolio_value > (context.position + 0.02):
                order(context.asset, -15, context.price[-1] - 0.01)
                order(context.asset, -10, context.price[-1] - 0.02)
                order(context.asset, -5, context.price[-1] - 0.03)

        #determine the position size
        if bull == True:
            targetpos = 1
        elif bear == True:
            targetpos = -1
        else:
            targetpos = 0
        #adjust position size
        if targetpos != 0:
            #
            if targetpos < 0:
                targetpos *= 0.5
            if abs(context.vol) < context.volthreshold:
                targetpos *= abs(context.vol) / context.volthreshold
            #
            if context.timelap < 10:
                targetpos *= 0.8
            if context.timelap < 20:
                targetpos *= 0.8
            if targetpos > 0:
                targetpos *= context.techsignal
            else:
                targetpos *= (1.8 - context.techsignal)
        #execute order
        
        if targetpos > 0.05:
           
            order(context.asset, min(targetpos * context.portfolio.portfolio_value / orderprice, abs((context.portfolio.portfolio_value - abs(context.portfolio.capital_used))/ orderprice), orderprice))
            #print context.price, '+'
            context.timelap = 0
            context.time = 0
        elif targetpos < -0.05:
            
            order(context.asset, -min(abs(targetpos * context.portfolio.portfolio_value) / orderprice, abs((context.portfolio.portfolio_value - abs(context.portfolio.capital_used)) / orderprice)) + 1, orderprice)
            
            print context.price, '-'
            context.timelap = 0
            context.time = 0
        else:
            context.timelap += 1
        #scan redundant order
        if context.time > 5:
            orders = get_open_orders(context.asset)
            if orders:
                for oo in orders:  
                    cancel_order(oo)  
            context.time = 0
    else:
        context.count += 1
