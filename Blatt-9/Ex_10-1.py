 
def PrintScatter(self,train):
	value = self.test_var
	predictS = train.predict(self.test_Signal)	
	predictB = train.predict(self.test_Background)	
	#print 'predict Signal = ', predictS
	#print 'predict Background = ', predictB 

	#check if prediction for Signal is correct or not
	for S in range(len(predictS)):
		if (predictS[S] != 1):
		  predictS[S] = 2	#Signal wrong classified
	for B in range(len(predictB)):
		if (predictB[B] != 0):
		  predictB[B] = -1	#Background wrong classified
	
	predict = np.concatenate((predictS, predictB))

	#plot for every combination of variables
	for pair, index in zip(self.varpairs, self.varindex):
		fig, ax = plt.subplots(figsize=(10,8))

		#compute ax-limits for scatterplots
		if min(value[:,index[0]]) < 0:
			low_x = min(value[:,index[0]])*1.05
		else:
			low_x = min(value[:,index[0]])*0.95
		if max(value[:,index[0]]) < 0:
			high_x = max(value[:,index[0]])*0.95
		else:
			high_x = max(value[:,index[0]])*1.055
		if min(value[:,index[1]]) < 0:
			low_y = min(value[:,index[1]])*1.05
		else:
			low_y = min(value[:,index[1]])*0.95
		if max(value[:,index[1]]) < 0:
			high_y = max(value[:,index[1]])*0.95
		else:
			high_y = max(value[:,index[1]])*1.055
		ax.set_xlim([low_x, high_x])
		ax.set_ylim([low_y, high_y])
		ax.set_xlabel(pair[0])
		ax.set_ylabel(pair[1])

		#create lists with correct/wrong classified Signal/Background
		wsx, wsy, wbx, wby, sx, sy, bx, by = [],[],[],[],[],[],[],[]
		for i in range(len(predict)):
			if (predict[i] == -1):
				wby.append(value[i,index[1]])
				wbx.append(value[i,index[0]])		#Background wrong classified
			elif (predict[i] == 2):
				wsy.append(value[i,index[1]])
				wsx.append(value[i,index[0]])		#Signal wrong classified
			elif (predict[i] == 1):
				sy.append(value[i,index[1]])
				sx.append(value[i,index[0]])		#Signal correct classified
			elif (predict[i] == 0):
				by.append(value[i,index[1]])
				bx.append(value[i,index[0]])		#Background correct classified
			else:
				print "Whuaaaat??? - wrong classification in "+str(i)

		ar1 = value[:,index[0]]
		ar2 = value[:,index[1]]
		norm = len(ar1)/1000
		plt.scatter(ar1[::norm],ar2[::norm], c=predict[::norm], cmap='rainbow', alpha=1)
		#plt.scatter(value[:,index[0]::1000], value[:,index[1]::1000], c=predict, cmap='rainbow', alpha = 1)
		plt.colorbar()
		plt.title('BDT prediction for 1000 Testevents ('+str(self.CLFname)+')')
		axes = fig.gca()
		ymin, ymax = axes.get_ylim()
		if low_x<0:
			xmark = low_x*1.07
		else:
			xmark = low_x*0.93
		if ymax>0:
			ymark = ymax*1.17
		else:
			ymark = ymax*0.83
		plt.text(xmark, ymark, train, verticalalignment='top', horizontalalignment='left', fontsize=7 )
		#plt.text(xmark, ymark, self.ReturnOpts(), verticalalignment='top', horizontalalignment='left', fontsize=7 )
		self.listoffigures.append(fig)
plt.close()