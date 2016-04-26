TGraphErrors* make_band(TF1& f, TMatrixD cov){
 double e1=f.GetParError(0);
 double e2=f.GetParError(1);
 double p1=f.GetParameter(0);
 double p2=f.GetParameter(1);
 double xmin=f.GetXmin();
 double xmax=f.GetXmax();
 int range=xmax-xmin;
 double ye[range];
 double y[range];
 double x[range];
// if (cov != NULL){
  for(int i=0; i<range; i++){
   x[i]=xmin+i;
   y[i]=p1*x[i]+p2;
   ye[i]=sqrt(cov(0,0)+(x[i]*x[i]*(cov(1,1)))+(((cov(0,1))*x[i])+(x[i]*(cov(1,0)))));
   cout<<ye[i]<<endl;
   cout<<(e1*e1)+(x[i]*x[i]*e2*e2)<<sqrt((e1*e1)+(x[i]*x[i]*e2*e2))<<ye[i]<<endl;
  }
// }
// else{
//  for(int i=0; i<range; i++){
//   x[i]=xmin+i;
//   y[i]=p1*x[i]+p2;
//   ye[i]=sqrt((e1*e1)+(x[i]*x[i]*e2*e2));
//   cout<<(e1*e1)+(x[i]*x[i]*e2*e2)<<sqrt((e1*e1)+(x[i]*x[i]*e2*e2))<<ye[i]<<endl;
//  }
// }
 TGraphErrors* g = new TGraphErrors(range, x, y, 0, ye);
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
 if (yerr[i]<0) yerr[i]=-yerr[i];
 cout<<yerr[i]<<endl;
}
cout<<xmin<<xmax<<endl;
TF1* f = new TF1("f", "[0]*x+[1]", xmin, xmax);
TGraphErrors* gr = new TGraphErrors(npoints, x, y, 0, yerr);
gr->SetTitle("TGraphErrors Example");
gr->SetMarkerColor(4);
gr->Fit("f","V");
gr->Draw("ALP");

TVirtualFitter* fitrp = TVirtualFitter::GetFitter();
Int_t nPar = fitrp->GetNumberTotalParameters();
TMatrixD covmat(nPar,nPar, fitrp->GetCovarianceMatrix());
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

TGraphErrors* g = make_band(*f, covmat);
g->SetFillStyle(3001);
g->Draw("A3 SAME");
gr->Draw("SAME");
c->Update();

fitrp->PrintResults(2,0);
}
