double wrong_pred(double P_rain, double P_sun_r, double P_rain_r)
{
double P_pred = ((P_rain)*(1-P_rain_r))+((1-P_rain)*(1-P_sun_r));
return P_pred;
}

void wrong_decision(double P_rain, double P_wrong, double P_sun_r, double P_rain_r)
{
double P_wet=0.;
P_wet = ((1-P_rain_r)*(P_rain))/(((1-P_rain_r)*P_rain)+(P_sun_r*(1-P_rain)));
double P_useless=0.;
P_useless = ((1-P_sun_r)*(1-P_rain))/((P_rain*P_rain_r)+((1-P_sun_r)*(1-P_rain)));
cout<<"Die Wahrscheinlichkeit nass zu werden ist: "<<P_wet<<endl;
cout<<"Die Wahrscheinlichkeit unnötig einen Schirm mitzunehmen ist :"<<P_useless<<endl;
}



void weather(){

double suncity = 0.05;		//probability of rain in suncity	P(rain)
double equalcity = 0.5;		//probability of rain in equalcity	P(rain)
double raincity = 0.95;		//probability of rain in raincity	P(rain)
//P(sun)=1-P(rain)
double sun_right = 0.9;		//sun predicted and correct		P(pred|sun)
double rain_right = 0.8;	//rain predicted and correct		P(pred|rain)
//P(pred) --> prediction correct
//P(sun|pred) --> sun shines if pred is correct
//P(rain|pred) --> rain if pred is correct
//wrong decision: 1-P(sun|pred)/1-P(rain|pred)

double wrong_pred_sun = 0;
wrong_pred_sun = wrong_pred(suncity, sun_right, rain_right);
cout<<"Falsche Vorhersage suncity: "<<wrong_pred_sun<<endl;
double wrong_pred_equal = 0;
wrong_pred_equal = wrong_pred(equalcity, sun_right, rain_right);
cout<<"Falsche Vorhersage equalcity: "<<wrong_pred_equal<<endl;
double wrong_pred_rain = 0;
wrong_pred_rain = wrong_pred(raincity, sun_right, rain_right);
cout<<"Falsche Vorhersage raincity: "<<wrong_pred_rain<<endl;

cout<<"In suncity:"<<endl;
wrong_decision(suncity, wrong_pred_sun, sun_right, rain_right);
cout<<"In equalcity:"<<endl;
wrong_decision(equalcity, wrong_pred_equal, sun_right, rain_right);
cout<<"In raincity:"<<endl;
wrong_decision(raincity, wrong_pred_rain, sun_right, rain_right);

cout << "test rechnung: " << (1-0.9)*0.05/((1-0.9)*0.05+(1-0.8)*(1-0.05)) << endl;
}

void MC_weather(int city_ID, int N)
{
double rain_prob = 1;
double rain = 0;
double pred = 0;
double rain_pred = 0.8;
double sun_pred = 0.9;
if (city_ID == 0) rain_prob = 0.05;
else if (city_ID == 1) rain_prob = 0.5;
else if (city_ID == 2) rain_prob = 0.95;
else cout<<"wrong city, 100% rain"<<endl;

 TRandom3 *r = new TRandom3;
 TCanvas* c = new TCanvas("c","c",800,600);
 TH1F *hist1 = new TH1F("hist1", "hist1", 4, 0, 4);
 TH1F *hist2 = new TH1F("hist2", "hist2", 2, 0, 4);

for (int i=0; i<N; i++){
 rain = r->Rndm();	//rain<rain_prob-->it rains
 pred = r->Rndm();	//pred<rain_pred-->correct if rain
 if (rain<rain_prob && pred<rain_pred){ hist1->Fill(0); hist2->Fill(0);}//regen, rergen vorhergesagt
 else if (rain<rain_prob && pred>=rain_pred){ hist1->Fill(1); hist2->Fill(3);}//regen, 
 else if (rain>=rain_prob && pred<sun_pred){ hist1->Fill(2); hist2->Fill(0);}
 else if (rain>=rain_prob && pred>=sun_pred){ hist1->Fill(3); hist2->Fill(3);}
}

hist1->GetXaxis()->SetBinLabel(1,"rain correct predicted");
hist1->GetXaxis()->SetBinLabel(2,"rain wrong predicted");
hist1->GetXaxis()->SetBinLabel(3,"sun correct predicted");
hist1->GetXaxis()->SetBinLabel(4,"sun wrong predicted");
//hist1->Scale(1/N);

hist2->GetXaxis()->SetBinLabel(1,"correct predicted");
hist2->GetXaxis()->SetBinLabel(2,"wrong predicted");

hist1->Draw("HIST");
//hist2->Draw("SAME HIST");

cout<<"Die Wahrscheinlichkeit nass zu werden ist: "<<hist1->GetBinContent(2)/N<<endl;
cout<<"Die Wahrscheinlichkeit unnötig einen Schirm mitzunehmen ist :"<<hist1->GetBinContent(4)/N<<endl;

cout<<"falsch Vorhergesagt: "<<hist2->GetBinContent(2)/N<<endl;
}
