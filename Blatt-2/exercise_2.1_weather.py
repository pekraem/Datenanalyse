import numpy as np

def pa(city_name, sun_rain,right_wrong): # vorhersage ist flasch
	if sun_rain == 0:
		pa = globals()[str("forcast_rain_"+right_wrong)] * globals()[str(city_name+"_city_rain")] / (globals()[str("forcast_rain_"+right_wrong)] * globals()[str(city_name+"_city_rain")] + (globals()[str("forcast_rain_"+right_wrong)] * globals()[str(city_name+"_city_sun")]))
	if sun_rain == 1:
		pa = globals()[str("forcast_sun_"+right_wrong)] * globals()[str(city_name+"_city_sun")] / (globals()[str("forcast_sun_"+right_wrong) ] * globals()[str(city_name+"_city_sun")] + (globals()[str("forcast_sun_"+right_wrong)] * globals()[str(city_name+"_city_rain")]))
	return pa

def pb(city_name, sun_rain):
	if sun_rain == 0:
		pb = globals()[str(city_name+"_city_sun")] * forcast_sun_wrong
	if sun_rain == 1:
		pb = globals()[str(city_name+"_city_rain")] * forcast_rain_wrong
	return pb
      
      
def p_ab(p_a):
	p_ba = 0.8
	p_nbna = 0.9
	p_bna = 1 - p_nbna
	p_nba = 1 - p_ba
	p_na = 1 - p_a
	p_b = p_a * p_ba + p_na * p_bna
	return p_ba * p_a/p_b
      
global sun_city_sun
sun_city_sun = 0.95
global sun_city_rain
sun_city_rain = 0.05

global equ_city_sun
equ_city_sun = 0.5
global equ_city_rain
equ_city_rain = 0.5

global rain_city_sun
rain_city_sun = 0.05
global rain_city_rain
rain_city_rain = 0.95

global forcast_sun_right
forcast_sun_right = 0.9
global forcast_sun_wrong
forcast_sun_wrong = 0.1

global forcast_rain_right
forcast_rain_right = 0.8
global forcast_rain_wrong
forcast_rain_wrong = 0.2


a1 = p_ab(0.05)
#a2 = pa("sun",0,"wrong")
#a3 = pa("sun",1,"right")
#a4 = pa("sun",1,"wrong")




print ("sun_city regen richtig: " + str(a1))

N = input("N: ")
rain_numbers = []
city_numbers = []
for i in range(N):
	rain_numbers.append(np.random.uniform(0, 1))
	city_numbers.append(np.random.uniform(0, 1))




def counter_city(rain_numbers, city_name, city_number):
	globals()[str(city_name+"_monte_rain_richtig")] = 0 
	globals()[str(city_name+"_monte_sun_richtig")] = 0
	globals()[str(city_name+"_monte_rain_falsch")] = 0
	globals()[str(city_name+"_monte_sun_falsch")] = 0
	limit_rain = globals()[str(city_name+"_city_rain")] # entspricht p_a
	p_b = 0
	for i in range(len(rain_numbers)):
		if city_number[i] < limit_rain:
		   
		   if rain_numbers[i] < 0.8:
		     p_b += 1
		     globals()[str(city_name+"_monte_rain_richtig")] += 1 # entspricht p_ba
		   else:
		     globals()[str(city_name+"_monte_rain_falsch")] += 1
		else:   
		  if rain_numbers[i] < 0.9:
		    globals()[str(city_name+"_monte_sun_richtig")] += 1
		  else:
		    p_b += 1
		    globals()[str(city_name+"_monte_sun_falsch")] += 1
	
	
	p_a = limit_rain
	p_ba = globals()[str(city_name+"_monte_rain_richtig")]/float(p_b)
	p_b = p_b / float(N) # entspricht p_b
	print p_ba
	print p_ba * p_a / p_b
	print globals()[str(city_name+"_monte_sun_falsch")]/float(N)*limit_rain/p_b
	print globals()[str(city_name+"_monte_sun_richtig")]/float(N)*limit_rain/p_b
	print globals()[str(city_name+"_monte_rain_falsch")]/float(N)*limit_rain/p_b
	print globals()[str(city_name+"_monte_rain_richtig")]/float(N)*limit_rain/p_b
	

counter_city(rain_numbers, "sun", city_numbers)

#a1 = pb("sun",0)
#a2 = pb("rain",0)
#a3 = pb("equ",0)

#b1 = pb("sun",1)
#b2 = pb("rain",1)
#b3 = pb("equ",1)


#print ("Monte Schirm aber Sonne in sun_city: " + str(a1))
#print ("Monte Schirm aber Sonne in rain_city: " + str(a2))
#print ("Monte Schirm aber Sonne in equ_city: " + str(a3))

#print ("Monte Kein Schirm aber Regen in sun_city: " + str(b1))
#print ("Monte Kein Schirm aber Regen in rain_city: " + str(b2))
#print ("Monte Kein Schirm aber Regen in equ_city: " + str(b3))

print ("------------------------------------------------------")

a1 = pa("sun",0)
a2 = pa("rain",0)
a3 = pa("equ",0)

b1 = pa("sun",1)
b2 = pa("rain",1)
b3 = pa("equ",1)


print ("Monte Schirm aber Sonne in sun_city: " + str(a1))
print ("Monte Schirm aber Sonne in rain_city: " + str(a2))
print ("Monte Schirm aber Sonne in equ_city: " + str(a3))

print ("Monte Kein Schirm aber Regen in sun_city: " + str(b1))
print ("Monte Kein Schirm aber Regen in rain_city: " + str(b2))
print ("Monte Kein Schirm aber Regen in equ_city: " + str(b3))
		

