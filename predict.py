import pandas as pd
import os
import numpy as np
import matplotlib.pylab as plt
from statsmodels.tsa.stattools import arma_order_select_ic
from statsmodels.tsa.arima_model import ARMA
import pyGPs
import sys
import glob
from copy import deepcopy

sep = os.path.sep

outPath = os.path.dirname(os.path.abspath(__file__))\
		 + sep + "output"

inPath = os.path.dirname(os.path.abspath(__file__))\
		 + sep + "dowjones.txt"

def getDateList():
	dataDate = []
	dataLines = open(inPath, "r").readlines()
	for line in dataLines:
		token = line.split(",")
		dataDate.append(token[0])
	return dataDate

def getInput(pivoteDate = "9999-99-99", minDate = "2015-07-06"):
	dates = []
	prices = []
	lines = open(inPath, "r").readlines()
	for line in lines:
		token = line.split(",")
		if (token[0] >= minDate and token[0] < pivoteDate):
			dates.append(token[0])
			prices.append(float(token[1]))
		if token[0] == pivoteDate:
			realPrice = float(token[1])
	lastPrice = prices[-1]
	timeIndex = pd.DatetimeIndex(dates)
	data = pd.Series(prices, index = timeIndex)
	return data, realPrice, lastPrice

def getDiffData(data, timeDiff = 1):
	dataToReturn = data
	for i in range(timeDiff):
		dataToReturn = pd.Series.diff(dataToReturn)
	return dataToReturn[1:]

def getReturnValueFromDiff(predictValue, diffDataList):
	value = predictValue
	for i in range(len(diffDataList) - 1):
		value += diffDataList[i][0] + diffDataList[i+1].sum()
	return value

def makeInputSequence(len):
	return np.array(list(range(1, len + 1)))

def drawResult(predictResults, realPrices, title):
	fig, ax = plt.subplots(figsize=(20,10))
	lw = 2
	plt.plot(np.arange(len(realPrices)), predictResults, color='darkorange', label = 'predict prices')
	plt.plot(np.arange(len(realPrices)), realPrices, color='navy', lw=lw, label = 'real prices')
	plt.xlabel('dates')
	plt.ylabel('target')
	plt.title(title)
	plt.legend()
	plt.savefig("Plots/" + title + ".png")

def update(date, correct, realPrice, lastPrice, predictPrice, method):
	if not os.path.exists(outPath):
		os.makedirs(outPath)
	columns = ["method", "Date", "realPrice", "lastPrice", "predictPrice", "correct"]
	outFile =  outPath + sep + "%s %s.txt"%(method, date)
	open(outFile, "w").write("%s\n"%(",".join(columns)))
	open(outFile, "a").write("%s,%s,%s,%s,%s,%s"\
		%(method, date, realPrice, lastPrice, predictPrice, correct))
	return True

def loadResultFromFile(method):
	predictItems = []
	for file in glob.glob(outPath + sep + method + " *.txt"):
		lines = open(file, 'r').readlines()
		headers = lines[0].split(",")
		tokens = lines[1].split(",")
		item = {}
		for j in range(len(headers)):
			item[headers[j]] = tokens[j]
		predictItems.append(item)
	predictItems.sort(key=lambda x:x['Date'])
	finalResults = []
	trendResults = []
	realPriceList = []
	for item in predictItems:
		finalResults.append(float(item['predictPrice']))
		trendResults.append(int(item['correct\n']))
		realPriceList.append(float(item['realPrice']))
	finalResults = np.array(finalResults)
	trendResults = np.array(trendResults)
	realPriceList = np.array(realPriceList)
	return finalResults, trendResults, realPriceList

def getComponents(data):
	trend = deepcopy(data[5:])
	for i in range(len(trend)):
		trend[i] = data[i:i+5].describe()[1]
	deTrendSeries = data[5:]/trend
	temprate0 = []
	temprate1 = []
	temprate2 = []
	temprate3 = []
	temprate4 = []
	for i in range(len(deTrendSeries)):
		if deTrendSeries.index[i].weekday() == 0:
			temprate0.append(deTrendSeries[i])
		if deTrendSeries.index[i].weekday() == 1:
			temprate1.append(deTrendSeries[i])
		if deTrendSeries.index[i].weekday() == 2:
			temprate2.append(deTrendSeries[i])
		if deTrendSeries.index[i].weekday() == 3:
			temprate3.append(deTrendSeries[i])
		if deTrendSeries.index[i].weekday() == 4:
			temprate4.append(deTrendSeries[i])

	seasonalComponents = np.array([np.mean(temprate0), np.mean(temprate1), np.mean(temprate2), np.mean(temprate3), np.mean(temprate4)])
	seasonal = deepcopy(data[5:])
	for i in range(len(seasonal)):
		if seasonal.index[i].weekday() == 0:
			seasonal[i] = seasonalComponents[0]
		if seasonal.index[i].weekday() == 1:
			seasonal[i] = seasonalComponents[1]
		if seasonal.index[i].weekday() == 2:
			seasonal[i] = seasonalComponents[2]
		if seasonal.index[i].weekday() == 3:
			seasonal[i] = seasonalComponents[3]
		if seasonal.index[i].weekday() == 4:
			seasonal[i] = seasonalComponents[4]

	residual = deTrendSeries/seasonal
	return trend, seasonal, residual

def armaPrediction(data):
	icOrder = arma_order_select_ic(data,
									ic = ['aic'],
									trend = 'c',
									max_ar = 5,
									max_ma = 6,
									fit_kw = {'method': 'mle'})
	aicOrder = icOrder["aic_min_order"]
	model = ARMA(data, order=aicOrder)
	results = model.fit(trend='c', method='mle', disp=-1)
	predictResult, sigmaResult, CI = results.forecast(steps = 1)
	return predictResult[0]

def gprPrediction(data):
	model = pyGPs.GPR()
	x = makeInputSequence(len(data))
	xt = np.array([x[-1] + 1])
	model.setData(x, data)
	k1 = pyGPs.cov.RBF()
	model.setPrior(kernel=k1)
	model.setOptimizer("Minimize", num_restarts = 30)
	model.optimize(x, data)
	ym = model.predict(xt)[0]
	predict = ym[0][0]
	return predict

def searchUpdate(finalMaxResult, finalMaxDate, finalMaxi, finalMaxj, listResults):
	if not os.path.exists(outPath):
		os.makedirs(outPath)
	columns = ["Date", "finalMaxResult", "finalMaxi", "finalMaxj", "listResults"]
	outFile =  outPath + sep + "search GPR-ARMA.txt"
	open(outFile, "w").write("%s\n"%(",".join(columns)))
	open(outFile, "a").write("%s,%s,%s,%s,%s"\
		%(finalMaxDate, finalMaxResult, finalMaxi, finalMaxj, listResults))
	return True

if __name__ == "__main__":
	def checkArgv(name):
		for a in sys.argv:
			if name == a:
				return True
		return False
	allDate = getDateList()
	firstDate = "2015-04-01"

	if checkArgv("search"):
		finalMaxResult = 0
		listResults = []
		while True:
			# we just search in range 01/04 to 11/11
			print("Start searching for best result, the fistDate is: %s" %(firstDate))
			if firstDate >= "2015-06-10":
				break

			for i in range(len(allDate)):
				if allDate[i] >= firstDate:
					predictDate = allDate[i+390:i+510]
					break

			finalResults = []
			trendResults = []
			realPriceList = []
			for date in predictDate:
				print("firstDate is %s" %(firstDate))
				# get data, realPrice, lastPrice as the input
				data, realPrice, lastPrice = getInput(pivoteDate = date, minDate = firstDate)
				# get components from series
				trend, seasonal, residual = getComponents(data)
				diffResidual = residual
				diffResidualList = [diffResidual]
				# predict residual, if doesn't success, get diff and predict again
				while True:
					try:
						print("Start predict residual using ARMA for date: %s" %(date))
						result = armaPrediction(diffResidual)
					except:
						diffResidual = getDiffData(diffResidual)
						diffResidualList.append(diffResidual)
						continue
					break
				# we has diffed data, now we must return the actual value
				returnResidual = getReturnValueFromDiff(result, diffResidualList)
				diffTrend = trend
				diffTrendList = [trend]
				# predict trend, if doesn't success, get diff and predict again
				while True:
					try:
						print("Start predict trend using GPR for date: %s" %(date))
						result = gprPrediction(diffTrend)
					except:
						diffTrend = getDiffData(diffTrend)
						diffTrendList.append(diffTrend)
						continue
					break
				returnTrend = getReturnValueFromDiff(result, diffTrendList)
				# predict seasonal
				weekday = pd.datetime.strptime(date, '%Y-%m-%d').weekday()
				returnSesonal = seasonal[weekday]
				# get returnvalue = predict value 
				returnValue = returnTrend*returnSesonal*returnResidual

				# correct prediction = 1 if it predict right trend. and equal to 0 if it doesn't
				correct = 1
				if returnValue > lastPrice and realPrice > lastPrice:
					trendResults.append(1)
				elif returnValue < lastPrice and realPrice < lastPrice:
					trendResults.append(1)
				else:
					trendResults.append(0)
					correct = 0
				# update(date, correct, realPrice, lastPrice, returnValue, method = 'GPR-ARMA')
				finalResults.append(returnValue)
				realPriceList.append(realPrice)
				print("ARMA result for Date: %s, correctPredict: %s, predictPrice: %s" %(date, correct, returnValue))

			finalResults = np.array(finalResults)
			trendResults = np.array(trendResults)
			correctPercent = np.sum(trendResults)/float(len(trendResults))
			# drawResult(finalResults, realPriceList, title = 'DowJones-Index prediction Using GPR-ARMA')
			print("Number of predict day: %s" %(len(finalResults)))
			print("TrendPredict result: %s" %correctPercent)
			maxResult = 0
			for i in range(20):
				for j in range(20-i):
					correctIJ = 0
					for ele in trendResults[i:100+j+i]:
						correctIJ += ele
					if correctIJ/float(100+j) > maxResult:
						maxj = j
						maxi = i
						maxResult = correctIJ/float(100+j)

			listResults.append((maxResult, firstDate, maxj, maxi))
			print("maxResult: %s, maxj: %s, maxi: %s" %(maxResult, maxj, maxi))
			if maxResult > finalMaxResult:
				finalMaxResult = maxResult
				finalMaxDate = firstDate
				finalMaxi = maxi
				finalMaxj = maxj

			for i in range(len(allDate)):
				if allDate[i] >= firstDate:
					firstDate = allDate[i+6]
					break
		searchUpdate(finalMaxResult, finalMaxDate, finalMaxi, finalMaxj, listResults)

	if checkArgv("arma") or checkArgv("ALL"):
		finalResults = []
		trendResults = []
		realPriceList = []
		for i in range(len(allDate)):
				if allDate[i] >= firstDate:
					predictDate = allDate[i+390:i+510]
					break
		for date in predictDate:
			data, realPrice, lastPrice = getInput(pivoteDate = date, minDate = firstDate)
			diffData = data
			diffDataList = [data]
			count = 0
			while True:
				try:
					result = armaPrediction(diffData)
				except:
					count += 1
					diffData = getDiffData(diffData)
					diffDataList.append(diffData)
					continue
				break

			returnValue = getReturnValueFromDiff(result, diffDataList)
			correct = 1
			if returnValue > lastPrice and realPrice > lastPrice:
				trendResults.append(1)
			elif returnValue < lastPrice and realPrice < lastPrice:
				trendResults.append(1)
			else:
				trendResults.append(0)
				correct = 0
			update(date, correct, realPrice, lastPrice, returnValue, method = 'ARMA')
			finalResults.append(returnValue)
			realPriceList.append(realPrice)
			print("ARMA result for Date: %s correctPredict: %s" %(date, correct))
		finalResults = np.array(finalResults)
		trendResults = np.array(trendResults)
		correctPercent = np.sum(trendResults)/float(len(trendResults))
		drawResult(finalResults, realPriceList, title = 'DowJones-Index prediction Using ARMA')
		print("Number of predict day: %s" %(len(finalResults)))
		print("Number of correct predict: %s" %(np.sum(trendResults)))
		print("TrendPredict result: %s" %correctPercent)

	if checkArgv("gpr") or checkArgv("ALL"):
		finalResults = []
		trendResults = []
		realPriceList = []
		for i in range(len(allDate)):
			if allDate[i] >= firstDate:
				predictDate = allDate[i+390:i+510]
				break
		for date in predictDate:
			data, realPrice, lastPrice = getInput(pivoteDate = date, minDate = firstDate)
			diffData = data
			diffDataList = [data]
			count = 0
			while True:
				try:
					result = gprPrediction(diffData)
				except:
					count += 1
					diffData = getDiffData(diffData)
					diffDataList.append(diffData)
					continue
				break

			returnValue = getReturnValueFromDiff(result, diffDataList)
			correct = 1
			if returnValue > lastPrice and realPrice > lastPrice:
				trendResults.append(1)
			elif returnValue < lastPrice and realPrice < lastPrice:
				trendResults.append(1)
			else:
				trendResults.append(0)
				correct = 0
			update(date, correct, realPrice, lastPrice, returnValue, method = 'GPR')
			finalResults.append(returnValue)
			realPriceList.append(realPrice)
			print("GPR result for Date: %s correctPredict: %s" %(date, correct))



		finalResults = np.array(finalResults)
		trendResults = np.array(trendResults)
		correctPercent = np.sum(trendResults)/float(len(trendResults))
		drawResult(finalResults, realPriceList, title = 'DowJones-Index prediction Using GPR')
		print("Number of predict day: %s" %(len(finalResults)))
		print("Number of correct predict: %s" %(np.sum(trendResults)))
		print("TrendPredict result: %s" %correctPercent)

	if checkArgv("show_result_arma"):
		finalResults, trendResults, realPriceList = loadResultFromFile('ARMA')
		correctPercent = np.sum(trendResults)/float(len(trendResults))
		drawResult(finalResults, realPriceList, title = 'DowJones-Index prediction Using ARMA')
		print("Number of predict day: %s" %(len(finalResults)))
		print("Number of correct predict: %s" %(np.sum(trendResults)))
		print("TrendPredict result: %s" %correctPercent)

	if checkArgv("show_result_gpr"):
		finalResults, trendResults, realPriceList = loadResultFromFile('GPR')
		correctPercent = np.sum(trendResults)/float(len(trendResults))
		drawResult(finalResults, realPriceList, title = 'DowJones-Index prediction Using GPR')
		print("Number of predict day: %s" %(len(finalResults)))
		print("Number of correct predict: %s" %(np.sum(trendResults)))
		print("TrendPredict result: %s" %correctPercent)

	if checkArgv("show_result_gpr-arma"):
		finalResults, trendResults, realPriceList = loadResultFromFile('GPR-ARMA')
		correctPercent = np.sum(trendResults)/float(len(trendResults))
		drawResult(finalResults, realPriceList, title = 'DowJones-Index prediction Using GPR-ARMA')
		print("Number of predict day: %s" %(len(finalResults)))
		print("Number of correct predict: %s" %(np.sum(trendResults)))
		print("TrendPredict result: %s" %correctPercent)