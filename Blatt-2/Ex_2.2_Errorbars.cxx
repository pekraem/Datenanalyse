TGraphErrors* make_band(TF1& f, TMatrixD* cov=NULL){
 double e1=f.GetParError(0);
 double e2=f.GetParError(1);
 double p1=f.GetParameter(0);
 double p2=f.GetParameter(1);
 double xmin=f.GetXmin();
 double xmax=f.GetXmax();
 int range=xmax-xmin;
 double *ye = new double[range];
 double *y = new double[range];
 double *x = new double[range];

 hist = f.GetHistogram();

 if ((cov != NULL)){
  cout<<"with covariance"<<endl;
 cout<<"Covarianz 0 0 = " << cov->operator()(0,0);
 cout<<"Covarianz 1 0 = " << cov->operator()(1,0);
 cout<<"Covarianz 1 1 = " << cov->operator()(1,1);
  for(int i=0; i<range; i++){
   //y[i]=hist->GetBinEntry(i);
   x[i]=i + 10.0;
   y[i] = i + 10.0;
   //ye[i]=sqrt((e1*e1)+(/*x[i]*x[i]**/e2*e2)+2*cov->operator()(0,1));

   ye[i]=sqrt(cov->operator()(1,1)+(x[i]*x[i]*(cov->operator()(0,0)))+(((cov->operator()(0,1))*x[i])+(x[i]*(cov->operator()(1,0)))));
   //g->SetPointError(i,0,ye[i]);
   cout<<ye[i]<<endl;
   cout<<(e1*e1)+(x[i]*x[i]*e2*e2) << " " <<sqrt((e1*e1)+(x[i]*x[i]*e2*e2))<< " " <<ye[i] << " " << x[i] <<endl;
  }
 }
 else{
  for(int i=0; i<range; i++){
   x[i]=xmin+i;
   y[i]=p1*x[i]+p2;
   ye[i]=sqrt((e1*e1)+(x[i]*x[i]*e2*e2));
   cout<<(e1*e1)+(x[i]*x[i]*e2*e2)<<sqrt((e1*e1)+(x[i]*x[i]*e2*e2))<<ye[i]<<endl;
  }
 }
 TGraphErrors* g = new TGraphErrors(range, x, y, 0, ye);
 //TGraphErrors* g = new TGraphErrors(hist);
 return g;
}

void Errorbars()
{

int xof = 10;
int npoints = 10;
double Steigung=1;
double yAchsenabschnitt=0;

TCanvas* c = new TCanvas("c","c",800,640);

TRandom3* r = new TRandom3(0);
double ye[npoints];
double x[npoints];
double y[npoints];
double yerr[npoints];
int xmin=xof;
int xmax=xof;

for(int i=0; i<npoints; i++){
 x[i]=xof+i;
 if (x[i]<xmin)xmin=x[i];
 if (x[i]>xmax){xmax=x[i];cout<<"\n\n\n\n\n>>>>>>>>>>>>>>>>>>>>"<<xmax<<endl;}
 y[i]=Steigung*x[i]+yAchsenabschnitt;
 yerr[i]=r->Gaus(0.,0.5);
 y[i] = y[i]+yerr[i];
 ye[i] = 0.5;
 //if (yerr[i]<0) yerr[i]=-yerr[i];
 //cout<<yerr[i]<<endl;
}
cout<<xmin<<xmax<<endl;
TF1* f = new TF1("f", "[0]*x+[1]", xmin, xmax);
TGraphErrors* gr = new TGraphErrors(npoints, x, y, 0, ye);
gr->SetTitle("TGraphErrors Example");
gr->SetMarkerColor(4);
gr->Fit("f","V");
gr->Draw("ALP");

TVirtualFitter* fitrp = TVirtualFitter::GetFitter();
Int_t nPar = fitrp->GetNumberTotalParameters();
TMatrixD covmat(nPar,nPar, fitrp->GetCovarianceMatrix());
TMatrixD* cov = &covmat;
cout<<"\n\nThe Covariance Matrix is:";
covmat.Print();
TMatrixD cormat(covmat);
for (Int_t i=0; i<nPar;i++){
 for (Int_t j=0; j<nPar; j++){
  cormat(i,j)/=sqrt(covmat(i,i))*sqrt(covmat(j,j));
 }
}
cout<<"\nThe Correlation Matrix is:";
cormat.Print();
cout<<covmat(0,0)<<covmat(0,1)<<covmat(1,0)<<covmat(1,1)<<endl;
/*
TGraphErrors* g = make_band(*f);
g->SetFillStyle(3001);
g->Draw("A4");
f->Draw("SAME");
gr->Draw("SAME");
c->Update();
*/
gn = make_band(*f, cov);
gn->SetFillColor(kRed);
gn->SetFillStyle(3001);
gn->Draw("A3");
f->Draw("SAME");
gr->Draw("SAME");
c->Update();

fitrp->PrintResults(2,0);
}
