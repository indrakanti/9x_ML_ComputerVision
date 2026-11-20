#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
struct A{int h,w,k,s,e;double cb;std::vector<double> cw,pw,pb,wq,wk,wv;};
std::vector<double> pv(const std::string&t){std::vector<double>o;std::stringstream s(t);std::string x;while(std::getline(s,x,','))o.push_back(std::stod(x));return o;}
A load(const std::string&p){std::ifstream f(p);if(!f)throw std::runtime_error("open");std::unordered_map<std::string,std::string>v;std::string l;while(std::getline(f,l)){if(l.empty()||l[0]=='#')continue;auto q=l.find('=');v[l.substr(0,q)]=l.substr(q+1);}A a{std::stoi(v["image_h"]),std::stoi(v["image_w"]),std::stoi(v["kernel"]),std::stoi(v["stride"]),std::stoi(v["embed"]),std::stod(v["conv_bias"]),pv(v["conv_weight"]),pv(v["proj_weight"]),pv(v["proj_bias"]),pv(v["wq"]),pv(v["wk"]),pv(v["wv"])};return a;}
std::vector<double> img(){std::vector<double>x(16);for(int i=0;i<16;++i)x[static_cast<std::size_t>(i)]=i+1;return x;}
std::vector<double> mm(const std::vector<double>&x,const std::vector<double>&w,int d){std::vector<double>o(static_cast<std::size_t>(d));for(int r=0;r<d;++r)for(int c=0;c<d;++c)o[static_cast<std::size_t>(r)]+=w[static_cast<std::size_t>(r*d+c)]*x[static_cast<std::size_t>(c)];return o;}
std::vector<double> fw(const A&a,const std::vector<double>&x){int oh=(a.h-a.k)/a.s+1,ow=(a.w-a.k)/a.s+1;std::vector<std::vector<double>>t;for(int y=0;y<oh;++y)for(int z=0;z<ow;++z){double v=a.cb;for(int ky=0;ky<a.k;++ky)for(int kx=0;kx<a.k;++kx)v+=x[static_cast<std::size_t>((y*a.s+ky)*a.w+z*a.s+kx)]*a.cw[static_cast<std::size_t>(ky*a.k+kx)];std::vector<double>r;for(int d=0;d<a.e;++d)r.push_back(a.pb[static_cast<std::size_t>(d)]+a.pw[static_cast<std::size_t>(d)]*v);t.push_back(r);}std::vector<std::vector<double>>q,k,v;for(auto&r:t){q.push_back(mm(r,a.wq,a.e));k.push_back(mm(r,a.wk,a.e));v.push_back(mm(r,a.wv,a.e));}double scale=std::sqrt(static_cast<double>(a.e));std::vector<double>o;for(auto&qr:q){std::vector<double>s;for(auto&kr:k){double z=0;for(int d=0;d<a.e;++d)z+=qr[static_cast<std::size_t>(d)]*kr[static_cast<std::size_t>(d)];s.push_back(z/scale);}double m=s[0],sum=0;for(double z:s)if(z>m)m=z;for(double&z:s){z=std::exp(z-m);sum+=z;}for(double&z:s)z/=sum;for(int d=0;d<a.e;++d){double z=0;for(std::size_t i=0;i<v.size();++i)z+=s[i]*v[i][static_cast<std::size_t>(d)];o.push_back(z);}}return o;}
void js(const std::string&p){auto o=fw(load(p),img());std::cout<<std::setprecision(17)<<"{\"values\":[";for(std::size_t i=0;i<o.size();++i){if(i)std::cout<<",";std::cout<<o[i];}std::cout<<"]}\n";}
}
int main(int c,char**v){try{if(c==3&&std::string(v[1])=="--json"){js(v[2]);return 0;}if(c==3&&std::string(v[1])=="--self-test")return fw(load(v[2]),img()).size()==8U?0:1;return 0;}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
