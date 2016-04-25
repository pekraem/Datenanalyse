//TGraphErrors* make_band(TF1& f, TMatrixDSym* cov){



void Errorbars()
{

int xof = 10;
int npoints = 10;
double Steigung=1;
double yAchsenabschnitt=0;

TCanvas* c = new TCanvas("c","c",800,640);
TF1* f = new TF1("f", "[0]*x+[1]", 0, 100);
TRandom3* r = new TRandom3(0);

double x[npoints];
double y[npoints];
double yerr[npoints];

for(int i=0; i<npoints; i++){
 x[i]=xof+i;
 y[i]=Steigung*x[i]+yAchsenabschnitt;
 yerr[i]=r->Gaus(0.,0.5);
 if (yerr[i]<0) yerr[i]=-yerr[i];
 cout<<yerr[i]<<endl;
}
cout<<yerr<<endl;

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

//gr->make_band("f","cormat");
gr->DrawClone();

fitrp->PrintResults(2,0);
}
