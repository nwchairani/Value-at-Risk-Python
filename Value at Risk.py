#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Value at Risk (VAR)
@author: Novia Widya Chairani
"""
# Getting the data
# Import packages 
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import datetime as dt
import scipy.stats as sts

# Portfolio's stock list
# MMM=3M, ABT=Abbott, ACN=Accenture, ARE=Alexandria Real Estate, AMZN=Amazon
stocks = ['MMM','ABT','ACN','ARE','AMZN']
names = ['3M','Abbott','Accenture', 'Alexandria', 'Amazon']

# Downloading the data from Yahoo! Finance
def download_data(stocks):
	data = web.DataReader(stocks, data_source='yahoo',start='31/01/2017',end='31/01/2022')['Adj Close'] #only get the closing price
	data.columns = stocks #set column names equal to stocks list
	return data

#Calculating percentage
data = web.DataReader(stocks,data_source='yahoo',start='31/01/2017',end='31/01/2022')['Adj Close']   
data_change= data.pct_change()

#obtain 0.05 quantile on the daily retuns distribution
data_change.quantile(0.05)

log_returns = np.log(data) - np.log(data.shift(1))

#############################

#Q1

#A. Get the 0.05 quantile and 0.01 quantile on the log daily retuns distribution (empirical VaR)
#histogram of log returns and plot the VaR

#3M
var_MMM_5percent=log_returns['MMM'].quantile(0.05)
print('Historical VaR: %0.3f' % var_MMM_5percent)

var_MMM_1percent=log_returns['MMM'].quantile(0.01)
print('Historical VaR: %0.3f' % var_MMM_1percent)
#histogram of log returns and plot the VaR
plt.figure(figsize=(10,10))
plt.hist(log_returns['MMM'], bins=50)
plt.axvline(var_MMM_5percent,color='red', linestyle='dashed', linewidth=2)
plt.axvline(var_MMM_1percent,color='green', linestyle='dashed', linewidth=2)
#plt.text(var, 100, var.round(3),)
plt.annotate(var_MMM_5percent.round(3), xy=(var_MMM_5percent,100), xytext=(0.07, 150),
             arrowprops=dict(facecolor='red', shrink=0.1))
plt.annotate(var_MMM_1percent.round(3), xy=(var_MMM_1percent,100), xytext=(-0.15,150),
             arrowprops=dict(facecolor='red', shrink=0.1))

#ABBOTT
var_ABT_5percent=log_returns['ABT'].quantile(0.05)
print('Historical VaR: %0.3f' % var_ABT_5percent)

var_ABT_1percent=log_returns['ABT'].quantile(0.01)
print('Historical VaR: %0.3f' % var_ABT_1percent)
#histogram of log returns and plot the VaR
plt.figure(figsize=(10,10))
plt.hist(log_returns['ABT'], bins=50)
plt.axvline(var_ABT_5percent,color='red', linestyle='dashed', linewidth=2)
plt.axvline(var_ABT_1percent,color='green', linestyle='dashed', linewidth=2)
#plt.text(var, 100, var.round(3),)
plt.annotate(var_ABT_5percent.round(3), xy=(var_ABT_5percent,100), xytext=(0.07, 150),
             arrowprops=dict(facecolor='red', shrink=0.1))
plt.annotate(var_ABT_1percent.round(3), xy=(var_ABT_1percent,100), xytext=(-0.15,150),
             arrowprops=dict(facecolor='red', shrink=0.1))

#ACCENTURE
var_ACN_5percent=log_returns['ACN'].quantile(0.05)
print('Historical VaR: %0.3f' % var_ACN_5percent)

var_ACN_1percent=log_returns['ACN'].quantile(0.01)
print('Historical VaR: %0.3f' % var_ACN_1percent)
#histogram of log returns and plot the VaR
plt.figure(figsize=(10,10))
plt.hist(log_returns['ACN'], bins=50)
plt.axvline(var_ACN_5percent,color='red', linestyle='dashed', linewidth=2)
plt.axvline(var_ACN_1percent,color='green', linestyle='dashed', linewidth=2)
#plt.text(var, 100, var.round(3),)
plt.annotate(var_ACN_5percent.round(3), xy=(var_ACN_5percent,100), xytext=(0.07, 150),
             arrowprops=dict(facecolor='red', shrink=0.1))
plt.annotate(var_ACN_1percent.round(3), xy=(var_ACN_1percent,100), xytext=(-0.15,150),
             arrowprops=dict(facecolor='red', shrink=0.1))

#ALEXANDRIA
var_ARE_5percent=log_returns['ARE'].quantile(0.05)
print('Historical VaR: %0.3f' % var_ARE_5percent)

var_ARE_1percent=log_returns['ARE'].quantile(0.01)
print('Historical VaR: %0.3f' % var_ARE_1percent)
#histogram of log returns and plot the VaR
plt.figure(figsize=(10,10))
plt.hist(log_returns['ARE'], bins=50)
plt.axvline(var_ARE_5percent,color='red', linestyle='dashed', linewidth=2)
plt.axvline(var_ARE_1percent,color='green', linestyle='dashed', linewidth=2)
#plt.text(var, 100, var.round(3),)
plt.annotate(var_ARE_5percent.round(3), xy=(var_ARE_5percent,100), xytext=(0.07, 150),
             arrowprops=dict(facecolor='red', shrink=0.1))
plt.annotate(var_ARE_1percent.round(3), xy=(var_ARE_1percent,100), xytext=(-0.15,150),
             arrowprops=dict(facecolor='red', shrink=0.1))

#AMAZON
var_AMZN_5percent=log_returns['AMZN'].quantile(0.05)
print('Historical VaR: %0.3f' % var_AMZN_5percent)

var_AMZN_1percent=log_returns['AMZN'].quantile(0.01)
print('Historical VaR: %0.3f' % var_AMZN_1percent)
#histogram of log returns and plot the VaR
plt.figure(figsize=(10,10))
plt.hist(log_returns['AMZN'], bins=50)
plt.axvline(var_AMZN_5percent,color='red', linestyle='dashed', linewidth=2)
plt.axvline(var_AMZN_1percent,color='green', linestyle='dashed', linewidth=2)
#plt.text(var, 100, var.round(3),)
plt.annotate(var_AMZN_5percent.round(3), xy=(var_AMZN_5percent,100), xytext=(0.07, 150),
             arrowprops=dict(facecolor='red', shrink=0.1))
plt.annotate(var_AMZN_1percent.round(3), xy=(var_AMZN_1percent,100), xytext=(-0.15,150),
             arrowprops=dict(facecolor='red', shrink=0.1))

#we can assume daily returns to be normally distributed: mean and variance (standard deviation)
#can describe the process
#get the mean and standard deviation return
mu = np.mean(log_returns)

sigma = np.std(log_returns)

#############################

#B. GETTING 1 DAY VAR
z1=sts.norm.ppf(0.05)
z2=sts.norm.ppf(0.01)

#############################

#C. 60 DAYS HORIZON

#3M
Var_model_MMM_5percent=(sigma['MMM']*z1)
Var_model_MMM_1percent=(sigma['MMM']*z2)

print('1 day Value at risk is @95 percent level: %0.3f' % Var_model_MMM_5percent)
print('1 day Value at risk is @99 percent level: %0.3f' % Var_model_MMM_1percent)

#get the 60 day VaR
n=60 #days
Var60_model_MMM_5percent=(mu['MMM']*n)+((sigma['MMM']*np.sqrt(n))*z1)
Var60_model_MMM_1percent=(mu['MMM']*n)+((sigma['MMM']*np.sqrt(n))*z2)
print('Value of VAR with 60 days @95 percent level:%0.3f' % Var60_model_MMM_5percent)
print('Value of VAR with 60 days @99 percent level:%0.3f' % Var60_model_MMM_1percent)

plt.figure(figsize=(10,10))
plt.hist(log_returns['MMM'], bins=50)
plt.axvline(Var60_model_MMM_5percent,color='red', linestyle='dashed', linewidth=2)
plt.axvline(Var60_model_MMM_1percent,color='green', linestyle='dashed', linewidth=2)
#plt.text(var, 100, var.round(3),)
plt.annotate(Var60_model_MMM_5percent.round(3), xy=(Var60_model_MMM_5percent,100), xytext=(0.07, 150),
             arrowprops=dict(facecolor='red', shrink=0.1))
plt.annotate(Var60_model_MMM_1percent.round(3), xy=(Var60_model_MMM_1percent,100), xytext=(-0.15,150),
             arrowprops=dict(facecolor='red', shrink=0.1))

#ABBOTT
Var_model_ABT_5percent=(sigma['ABT']*z1)
Var_model_ABT_1percent=(sigma['ABT']*z2)

print('1 day Value at risk is @95 percent level: %0.3f' % Var_model_ABT_5percent)
print('1 day Value at risk is @99 percent level: %0.3f' % Var_model_ABT_1percent)

#get the 60 day VaR
n=60 #days
Var60_model_ABT_5percent=(mu['ABT']*n)+((sigma['ABT']*np.sqrt(n))*z1)
Var60_model_ABT_1percent=(mu['ABT']*n)+((sigma['ABT']*np.sqrt(n))*z2)
print('Value of VAR with 60 days @95 percent level:%0.3f' % Var60_model_ABT_5percent)
print('Value of VAR with 60 days @99 percent level:%0.3f' % Var60_model_ABT_1percent)

plt.figure(figsize=(10,10))
plt.hist(log_returns['ABT'], bins=50)
plt.axvline(Var60_model_ABT_5percent,color='red', linestyle='dashed', linewidth=2)
plt.axvline(Var60_model_ABT_1percent,color='green', linestyle='dashed', linewidth=2)
#plt.text(var, 100, var.round(3),)
plt.annotate(Var60_model_ABT_5percent.round(3), xy=(Var60_model_ABT_5percent,100), xytext=(0.07, 150),
             arrowprops=dict(facecolor='red', shrink=0.1))
plt.annotate(Var60_model_ABT_1percent.round(3), xy=(Var60_model_ABT_1percent,100), xytext=(-0.15,150),
             arrowprops=dict(facecolor='red', shrink=0.1))

#ACCENTURE
Var_model_ACN_5percent=(sigma['ACN']*z1)
Var_model_ACN_1percent=(sigma['ACN']*z2)

print('1 day Value at risk is @95 percent level: %0.3f' % Var_model_ACN_5percent)
print('1 day Value at risk is @99 percent level: %0.3f' % Var_model_ACN_1percent)

#get the 60 day VaR
n=60 #days
Var60_model_ACN_5percent=(mu['ACN']*n)+((sigma['ACN']*np.sqrt(n))*z1)
Var60_model_ACN_1percent=(mu['ACN']*n)+((sigma['ACN']*np.sqrt(n))*z2)
print('Value of VAR with 60 days @95 percent level:%0.3f' % Var60_model_ACN_5percent)
print('Value of VAR with 60 days @99 percent level:%0.3f' % Var60_model_ACN_1percent)

plt.figure(figsize=(10,10))
plt.hist(log_returns['ACN'], bins=50)
plt.axvline(Var60_model_ACN_5percent,color='red', linestyle='dashed', linewidth=2)
plt.axvline(Var60_model_ACN_1percent,color='green', linestyle='dashed', linewidth=2)
#plt.text(var, 100, var.round(3),)
plt.annotate(Var60_model_ACN_5percent.round(3), xy=(Var60_model_ACN_5percent,100), xytext=(0.07, 150),
             arrowprops=dict(facecolor='red', shrink=0.1))
plt.annotate(Var60_model_ACN_1percent.round(3), xy=(Var60_model_ACN_1percent,100), xytext=(-0.15,150),
             arrowprops=dict(facecolor='red', shrink=0.1))

#ALEXANDRIA
Var_model_ARE_5percent=(sigma['ARE']*z1)
Var_model_ARE_1percent=(sigma['ARE']*z2)

print('1 day Value at risk is @95 percent level: %0.3f' % Var_model_ARE_5percent)
print('1 day Value at risk is @99 percent level: %0.3f' % Var_model_ARE_1percent)

#get the 60 day VaR
n=60 #days
Var60_model_ARE_5percent=(mu['ARE']*n)+((sigma['ARE']*np.sqrt(n))*z1)
Var60_model_ARE_1percent=(mu['ARE']*n)+((sigma['ARE']*np.sqrt(n))*z2)
print('Value of VAR with 60 days @95 percent level:%0.3f' % Var60_model_ARE_5percent)
print('Value of VAR with 60 days @99 percent level:%0.3f' % Var60_model_ARE_1percent)

plt.figure(figsize=(10,10))
plt.hist(log_returns['ARE'], bins=50)
plt.axvline(Var60_model_ARE_5percent,color='red', linestyle='dashed', linewidth=2)
plt.axvline(Var60_model_ARE_1percent,color='green', linestyle='dashed', linewidth=2)
#plt.text(var, 100, var.round(3),)
plt.annotate(Var60_model_ARE_5percent.round(3), xy=(Var60_model_ARE_5percent,100), xytext=(0.07, 150),
             arrowprops=dict(facecolor='red', shrink=0.1))
plt.annotate(Var60_model_ARE_1percent.round(3), xy=(Var60_model_ARE_1percent,100), xytext=(-0.15,150),
             arrowprops=dict(facecolor='red', shrink=0.1))

#AMAZON
Var_model_AMZN_5percent=(sigma['AMZN']*z1)
Var_model_AMZN_1percent=(sigma['AMZN']*z2)

print('1 day Value at risk is @95 percent level: %0.3f' % Var_model_AMZN_5percent)
print('1 day Value at risk is @99 percent level: %0.3f' % Var_model_AMZN_1percent)

#get the 60 day VaR
n=60 #days
Var60_model_AMZN_5percent=(mu['AMZN']*n)+((sigma['AMZN']*np.sqrt(n))*z1)
Var60_model_AMZN_1percent=(mu['AMZN']*n)+((sigma['AMZN']*np.sqrt(n))*z2)
print('Value of VAR with 60 days @95 percent level:%0.3f' % Var60_model_AMZN_5percent)
print('Value of VAR with 60 days @99 percent level:%0.3f' % Var60_model_AMZN_1percent)

plt.figure(figsize=(10,10))
plt.hist(log_returns['AMZN'], bins=50)
plt.axvline(Var60_model_AMZN_5percent,color='red', linestyle='dashed', linewidth=2)
plt.axvline(Var60_model_AMZN_1percent,color='green', linestyle='dashed', linewidth=2)
#plt.text(var, 100, var.round(3),)
plt.annotate(Var60_model_AMZN_5percent.round(3), xy=(Var60_model_AMZN_5percent,100), xytext=(0.07, 150),
             arrowprops=dict(facecolor='red', shrink=0.1))
plt.annotate(Var60_model_AMZN_1percent.round(3), xy=(Var60_model_AMZN_1percent,100), xytext=(-0.15,150),
             arrowprops=dict(facecolor='red', shrink=0.1))

########################################################################

#Q2
#Portfolio VAR Computing
#choosing weights for OPTIMAL PORTFOLIO ()
weights_opt= np.array([0.016,0.297,0.397,0.022,0.268])
#daily expected portfolio return
portfolio_return_opt = np.sum(log_returns.mean()*weights_opt)
print("Expected daily portfolio return:", portfolio_return_opt)

#daily expected portfolio volatility
portfolio_volatility_opt = np.sqrt(np.dot(weights_opt.T, np.dot(log_returns.cov(),weights_opt)))
print("Expected daily volatility:", portfolio_volatility_opt)

#get the 1 day Var @ 95 percent level and 99 percent level

Var_opt_portfolio_95percent=(portfolio_volatility_opt*z1)
Var_opt_portfolio_99percent=(portfolio_volatility_opt*z2)
print('1 day Portfolio Value at risk @95 percent is: %0.3f' % Var_opt_portfolio_95percent)
print('1 day Portfolio Value at risk @99 percent is: %0.3f' %Var_opt_portfolio_99percent )

#get the 60 day Var
n=60 #days
Var60_opt_portfolio_95percent=(portfolio_return_opt*n)+((portfolio_volatility_opt*np.sqrt(n))*z1)
Var60_opt_portfolio_99percent=(portfolio_return_opt*n)+((portfolio_volatility_opt*np.sqrt(n))*z2)
print('60 days Portfolio Value at risk @95 percent is: %0.3f' % Var60_opt_portfolio_95percent)
print('60 days Portfolio Value at risk @99 percent is: %0.3f' % Var60_opt_portfolio_99percent)

#MINIMUM VARIANCE
weights_min=np.array([0.274,0.201,0.063,0.275,0.186])

#daily expected portfolio return
portfolio_min_return = np.sum(log_returns.mean()*weights_min)
print("Expected daily portfolio return:", portfolio_min_return)

#daily expected portfolio volatility
portfolio_min_volatility = np.sqrt(np.dot(weights_min.T, np.dot(log_returns.cov(),weights_min)))
print("Expected daily volatility:", portfolio_min_volatility)

#get the 1 day Var @ 95 percent level and @ 99 percent level
Var_min_portfolio_95percent=(portfolio_min_volatility*z1)
Var_min_portfolio_99percent=(portfolio_min_volatility*z2)
print('1 day Portfolio Value at risk @95 percent is: %0.3f' % Var_min_portfolio_95percent)
print('1 day Portfolio Value at risk @99 percent is: %0.3f' % Var_min_portfolio_99percent )

#get the 60 day Var
n=60 #days
Var60_min_portfolio_95percent=(portfolio_min_return*n)+((portfolio_min_volatility*np.sqrt(n))*z1)
Var60_min_portfolio_99percent=(portfolio_min_return*n)+((portfolio_min_volatility*np.sqrt(n))*z2)
print('60 days Portfolio Value at risk @95 percent is: %0.3f' % Var60_min_portfolio_95percent)
print('60 days Portfolio Value at risk @99 percent is: %0.3f' % Var60_min_portfolio_99percent)

########################################################################

#Q3
#plot data series
data.plot(figsize=(10,5), title='Stock Price Series')
plt.ylabel("Adjusted Closing Price")
plt.show()
























































