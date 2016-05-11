//Ex_3.1_Decay


double likelyhood(double* x, double theta){
  double likely = 1.;
  int i=0;
  do {
    likely = likely * x[i] * theta;
    cout << likely << endl;
    i++;
  }
  while(x[i]!=x[-1]);
  return likely;
}

double max_likelyhood(double* x, double theta_min, double theta_max, int steps){
  double besttheta = 0;
  double bestlikely = 0;
  double l = 0;
  for (double i = theta_min; i<theta_max; i+((theta_max-theta_min)/steps)){
    l = likelyhood(x,i);
    if (l>bestlikely){
      bestlikely = l;
      besttheta = i;
    }
  }
  return besttheta;
}
  

void expdecay(){
  //generate 10000 Randomnumbers and apply transformation method
  const int nbins = 100;
  TRandom3* r = new TRandom3(0);
  TCanvas* c = new TCanvas("c","c",800,600);
  TH1F* h = new TH1F("h","h", nbins, 0 , 10);
  //T1F* L = new TF1("L", "L", nbins, 0, 10);
  const int N = 10000;
  double tau = 1.;
//   double f = 0.;
//   double likelyhood(double* likely, double* x, double* par){
//     double likely = 1.;
//     //do {
//     //  likely = likely * x[i] * par[0];}
//     //while(x[i]!=x[-1]);
//     return likely * x[i] * par[0];}
//   double sum = 0;
  
  for( int i = 0; i < N; i++ ){
    double tmp = r->Rndm();
    double x = log(tau/tmp);
    h->Fill(x);
  }
  double norm = 1./(h->Integral());
  //h->Scale(norm);
  //h->Fit("likelyhood");
  //h->Fit("l");
  //h->Draw("HIST");
  
  double x[nbins];
  for (int i = 0; i<nbins; i++){
    x[i] = h->GetBinContent(i);
    cout << x[i] << endl;
  }
  
  double besttheta = 0;
  besttheta = max_likelyhood(&x, 0.9, 1.1, 20);
  
  cout << "Likelyhood Fit = " << besttheta << endl;
//   double like = 0.;
//   like = likelyhood(&x,1.);
//   cout << "Likelyhood Fit = " << like << endl;



  TH1F* h1 = new TH1F("h1","h1", 100, 0 , 2);

  
  c->Close();
  
  for (int i=0; i<1000; i++){
    for (int k=0; k<10; k++){
      double tmp = r->Rndm();
      double x = log(tau/tmp);
      h1->Fill(x); 
    }
  }
}