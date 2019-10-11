
# coding: utf-8

# In[2]:


# later: calculate average daily deviation, and likelihood of making money
ticker = 'BABA'

import urllib2
from bs4 import BeautifulSoup
import certifi as cert
import urllib3 as url
quote_page = 'https://finance.yahoo.com/quote/' + ticker + '/options?p=' + ticker
page = urllib2.urlopen(quote_page)
soup = BeautifulSoup(page, 'html.parser')
call_strike = []
put_strike = []
call_bid = []
put_bid = []
call_ask = []
put_ask = []
call_tradable = []
put_tradable = []
last = 0.0
jud = 0
temp = soup.find_all('a',{'href':'/quote/' + ticker + '/options?strike=false&straddle=false','data-symbol':'' + ticker + ''})
temp1 = soup.find_all('td', {'class':'data-col4 Ta(end) Pstart(7px)'})
temp2 = soup.find_all('td', {'class':'data-col5 Ta(end) Pstart(7px)'})
for i in range(len(temp)):
    if jud == 0:
        if last < float(temp[i].get_text()):
            last = float(temp[i].get_text())
            call_strike.append(float(temp[i].get_text()))
            call_bid.append(float(temp1[i].get_text()))
            call_ask.append(float(temp2[i].get_text()))
            if float(temp1[i].get_text()) > 0.0 and float(temp2[i].get_text()) > 0.0:
                call_tradable.append(1)
            else:
                call_tradable.append(0)
        else:
            jud = 1
            put_strike.append(float(temp[i].get_text()))
            put_bid.append(float(temp1[i].get_text()))
            put_ask.append(float(temp2[i].get_text()))
            if float(temp1[i].get_text()) > 0.0 and float(temp2[i].get_text()) > 0.0:
                put_tradable.append(1)
            else:
                put_tradable.append(0)
    else:
        put_strike.append(float(temp[i].get_text()))
        put_bid.append(float(temp1[i].get_text()))
        put_ask.append(float(temp2[i].get_text()))
        if float(temp1[i].get_text()) > 0.0 and float(temp2[i].get_text()) > 0.0:
            put_tradable.append(1)
        else:
            put_tradable.append(0)

def get_stock_price(name):
    http = url.PoolManager(cert_reqs='CERT_REQUIRED', ca_certs=cert.where())
    html_doc = http.request('GET', 'https://finance.yahoo.com/quote/' + name + '?p=' + name)
    soup = BeautifulSoup(html_doc.data, 'html.parser')
    return soup.find("span", class_="Trsdu(0.3s) Fw(b) Fz(36px) Mb(-4px) D(ib)").get_text()
            
            

current = float(get_stock_price(ticker))

# safest straddle

best_range = 9999.9
best_strike = 0.0
rec1 = 0.0
for i in range(len(call_strike)):
    for j in range(len(put_strike)):
        if call_strike[i] == put_strike[j] and call_tradable[i] == 1 and put_tradable[j] == 1:
            price = 2 * call_ask[i] + 2 * put_ask[j] - call_bid[i] - put_bid[j]
            if abs(abs(call_strike[i] - current) - price) < best_range:
                best_range = abs(abs(call_strike[i] - current) - price)
                best_strike = call_strike[i]
                rec1 = price
print "All the data are from Yahoo Finance"
#print "The result below might be too conservative, because"
#print "1. out-of-money options are usually not worthless"
#print "2. smallest price range has been adjusted for bid-ask spread"
print "smallest range: smallest deviation from the current stock price in order to be profitable"
print "best strike: optimal strike price for call/put option"
print "price of option: sum of lowest ask price of call and put option with strike prices at 'best strike' and bid-ask spread"
print "-------------------------------------------------------------"
print "current price: ", current
print "smallest range: ", best_range
print "best strike: ", best_strike
print "price of option", rec1

