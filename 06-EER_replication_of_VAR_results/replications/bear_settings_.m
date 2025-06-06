%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                          %
%    BAYESIAN ESTIMATION, ANALYSIS AND REGRESSION (BEAR) TOOLBOX           %
%                                                                          %
%    This statistical package has been developed by the external           %
%    developments division of the European Central Bank.                   %
%                                                                          %
%    Authors:                                                              %
%    Alistair Dieppe (adieppe@ecb.europa.eu)                               %
%    Bj�rn van Roye  (Bjorn.van_Roye@ecb.europa.eu)                        %
%                                                                          %
%    Version 5.0                                                           %
%                                                                          %
%    The authors are grateful to the following people for valuable input   %
%    and advice which contributed to improve the quality of the toolbox:   %
%    Paolo Bonomolo, Mirco Balatti, Marta Banbura, Niccolo Battistini,     %
%	 Gabriel Bobeica, Martin Bruns, Fabio Canova, Matteo Ciccarelli,       %
%    Marek Jarocinski, Michele Lenza, Francesca Loria, Mirela Miescu,      %
%    Gary Koop, Chiara Osbat, Giorgio Primiceri, Martino Ricci,            %
%    Michal Rubaszek, Barbara Rossi, Ben Schumann, Marius Schulte,         %
%    Peter Welz and Hugo Vega de la Cruz. 						           %
%                                                                          %
%    These programmes are the responsibilities of the authors and not of   %
%    the ECB and all errors and ommissions remain those of the authors.    %
%                                                                          %
%    Using the BEAR toolbox implies acceptance of the End User Licence     %
%    Agreement and appropriate acknowledgement should be made.             %
%                                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% general data and model information

% VAR model selected (1=OLS VAR, 2=BVAR, 3=mean-adjusted BVAR, 4=panel Bayesian VAR, 5=Stochastic volatility BVAR, 6=Time varying)
VARtype=2;
% data frequency (1=yearly, 2= quarterly, 3=monthly, 4=weekly, 5=daily, 6=undated)
frequency=2;
% sample start date; must be a string consistent with the date formats of the toolbox
startdate='1974q1';
% sample end date; must be a string consistent with the date formats of the toolbox
enddate='2014q4';
% endogenous variables; must be a single string, with variable names separated by a space
varendo='DOM_GDP DOM_CPI STN';
% exogenous variables, if any; must be a single string, with variable names separated by a space
varexo='';
% number of lags
lags=4;
% inclusion of a constant (1=yes, 0=no)
const=1;
% path to data; must be a single string
cd ..\
pref.datapath=pwd; % main BEAR folder, specify otherwise
cd .\files
% excel results file name
pref.results_sub='results_bvr';
% to output results in excel
pref.results=1;
% output charts
pref.plot=1;
% pref: useless by itself, just here to avoid code to crash
pref.pref=0;
% save matlab workspace (1=yes, 0=no (default))
pref.workspace=0;

% FAVAR options
favar.FAVAR=0; % augment VAR model with factors (1=yes, 0=no)
    if favar.FAVAR==1
    % transform information variables in excel sheet 'factor data' (following Stock & Watson: 1 Level, 2 First Difference, 3 Second Difference, 4 Log-Level, 5 Log-First-Difference, 6 Log-Second-Difference)
    favar.transformation=0; % (1=yes, 0=no) // 'factor data' must contain values for startdate -1 in the case we have First Difference (2,5) transformation types and startdate -2 in the case we have Second Difference (3,6) transformation types
		favar.transform_endo=''; %transformation codes of varendo variables other than factors (ordering follows 'data' sheet!)
        
    % number of factors to include
    favar.numpc=3;
    
    % specify the ordering of endogenous factors and variables
    varendo='factor1 factor2 factor3 FYFF'; 
    
    % slow fast scheme for recursive identification (IRFt 2, 3) as in BBE (2005)
    favar.slowfast=1;  % assign variables in the excel sheet 'factor data' in the 'block' row to "slow" or "fast"
    
    % VARtype specific FAVAR options
    if VARtype==2 % supported priors: 1x, 2x, 3x, 41
    favar.onestep=1; % Bayesian estimation of factors and the model in an one-step estimation (1=yes, 0=no (two-step))   
    % thining of Gibbs draws
    favar.thin=1; % (=1 default, no thinning)
    % priors on factor equation
        % Loadings L~N(0,L0*eye)
        favar.L0=1; %BBE set-up
        % Covariance Sigma~IG(a,b)
        favar.a0=3; %BBE set-up
        favar.b0=0.001; %BBE set-up
    end
    
    % blocks/categories (1=yes, 0=no), specify in excel sheet
    favar.blocks=0;
        if favar.blocks==1 % assign information variables to blocks
            favar.blocknames='slow fast'; % specify in excel sheet 'factor data'
            favar.blocknumpc='2 2'; %block-specific number of factors (principal components)
        end
      
    % specify information variables of interest (IRF, FEVD, HD)
    favar.plotX='IP PUNEW FYGM3 FYGT5 FMFBA FM2 EXRJAN PMCP IPXMCA GMCQ GMCDQ GMCNQ LHUR PMEMP LEHCC HSFR PMNO FSDXP HHSNTN';
    
    % re-tranform transformed variables
    favar.levels=1; % =0 no re-transformation (default), =1 cumsum, =2 exp cumsum
    
    % (approximate) IRFs for information variables 
    favar.IRF.plot=1; % (1=yes, 0=no)   
    if favar.IRF.plot==1
        % choose shock(s) to plot
        favar.IRF.plotXshock='FYFF';
    end
    
    % (approximate) FEVDs for information variables
    favar.FEVD.plot=1; % (1=yes, 0=no)
    
    % (approximate) HDs for information variables
    favar.HD.plot=0; % (1=yes, 0=no)
    if favar.HD.plot==1
        favar.HD.sumShockcontributions=0; % sum contributions over shocks (origin ,=1), or over variables (impact, =0 (default))
        if favar.blocks==1 % plotting options
        favar.HD.plotXblocks=1; % sum contributions of factors blockwise
            favar.HD.HDallsumblock=1; % include all components of HDall(=1) other than shock contributions, but display them sumed under blocks\shocks (=0, default)
        end
    end
    end


% BVAR specific information: will be read only if VARtype=2

if VARtype==2
% selected prior
% 11=Minnesota (univariate AR), 12=Minnesota (diagonal VAR estimates), 13=Minnesota (full VAR estimates)
% 21=Normal-Wishart(S0 as univariate AR), 22=Normal-Wishart(S0 as identity)
% 31=Independent Normal-Wishart(S0 as univariate AR), 32=Independent Normal-Wishart(S0 as identity)
% 41=Normal-diffuse
% 51=Dummy observations
% 61=Mean-adjusted
prior=12;
% hyperparameter: autoregressive coefficient
ar=0.8;
% switch to Excel interface
PriorExcel=0; % set to 1 if you want individual priors, 0 for default
%switch to Excel interface for exogenous variables
priorsexogenous=0; % set to 1 if you want individual priors, 0 for default
% hyperparameter: lambda1
lambda1=0.1;
% hyperparameter: lambda2
lambda2=0.5;
% hyperparameter: lambda3
lambda3=1;
% hyperparameter: lambda4
lambda4=100;
% hyperparameter: lambda5
lambda5=0.001;
% hyperparameter: lambda6
lambda6=1;
% hyperparameter: lambda7
lambda7=0.1;
% Overall tightness on the long run prior
lambda8=1;
% (61=Mean-adjusted BVAR) Scale up the variance of the prior of factor f
priorf=100;
% total number of iterations for the Gibbs sampler
It=2000;
% number of burn-in iterations for the Gibbs sampler
Bu=1000;
% hyperparameter optimisation by grid search (1=yes, 0=no)
hogs=0;
% block exogeneity (1=yes, 0=no)
bex=0;
% sum-of-coefficients application (1=yes, 0=no)
scoeff=0;
% dummy initial observation application (1=yes, 0=no)
iobs=0;
% Long run prior option
lrp=0;
% create H matrix for the long run priors 
% now taken from excel loadH.m
% H=[1 1 0 0;-1 1 0 0;0 0 1 1;0 0 -1 1];


elseif VARtype==3 % Mean-adjusted BVAR specific information: will be read only if VARtype=3, (subsumed under VARtype 2 prior 61)


% panel Bayesian VAR specific information: will be read only if VARtype=4

elseif VARtype==4
% choice of panel model 
% 1=OLS mean group estimator, 2=pooled estimator
% 3=random effect (Zellner and Hong), 4=random effect (hierarchical)
% 5=static factor approach, 6=dynamic factor approach
panel=2;
% units; must be single sstring, with names separated by a space
unitnames='US EA UK';
% total number of iterations for the Gibbs sampler
It=2000;
% number of burn-in iterations for the Gibbs sampler
Bu=1000;
% choice of retaining only one post burn iteration over 'pickf' iterations (1=yes, 0=no)
pick=0;
% frequency of iteration picking (e.g. pickf=20 implies that only 1 out of 20 iterations will be retained)
pickf=20;
% hyperparameter: autoregressive coefficient
ar=0.8;
% hyperparameter: lambda1
lambda1=0.1;
% hyperparameter: lambda2
lambda2=0.5;
% hyperparameter: lambda3
lambda3=1;
% hyperparameter: lambda4
lambda4=100;
% hyperparameter: s0
s0=0.001;
% hyperparameter: v0
v0=0.001;
% hyperparameter: alpha0
alpha0=1000;
% hyperparameter: delta0
delta0=1;
% hyperparameter: gama
gama=0.85;
% hyperparameter: a0
a0=1000;
% hyperparameter: b0
b0=1;
% hyperparameter: rho
rho=0.75;
% hyperparameter: psi
psi=0.1;



% Stochastic volatility BVAR information: will be read only if VARtype=5

elseif VARtype==5
% choice of stochastic volatility model 
% 1=standard, 2=random scaling, 3=large BVAR
stvol=1;
% total number of iterations for the Gibbs sampler
It=2000;
% number of burn-in iterations for the Gibbs sampler
Bu=1000;
% choice of retaining only one post burn iteration over 'pickf' iterations (1=yes, 0=no)
pick=0;
% frequency of iteration picking (e.g. pickf=20 implies that only 1 out of 20 iterations will be retained)
pickf=20;
% block exogeneity (1=yes, 0=no)
bex=0;
% hyperparameter: autoregressive coefficient
ar=0.8;
% hyperparameter: lambda1
lambda1=0.1;
% hyperparameter: lambda2
lambda2=0.5;
% hyperparameter: lambda3
lambda3=1;
% hyperparameter: lambda4
lambda4=100;
% hyperparameter: lambda5
lambda5=0.001;
% hyperparameter: gama
gamma=0.85;
% hyperparameter: alpha0
alpha0=0.001;
% hyperparameter: delta0
delta0=0.001;
% hyperparameter: gamma0
gamma0=0;
% hyperparameter: zeta0
zeta0=10000;


% Time-varying BVAR information: will be read only if VARtype=6

elseif VARtype==6
% choice of time-varying BVAR model 
% 1=time-varying coefficients, 2=general time-varying
tvbvar=1;
% total number of iterations for the Gibbs sampler
It=2000;
% number of burn-in iterations for the Gibbs sampler
Bu=1000;
% choice of retaining only one post burn iteration over 'pickf' iterations (1=yes, 0=no)
pick=0;
% frequency of iteration picking (e.g. pickf=20 implies that only 1 out of 20 iterations will be retained)
pickf=20;
% calculate IRFs for every sample period (1=yes, 0=no)
alltirf=0;
% hyperparameter: gama
gamma=0.85;
% hyperparameter: alpha0
alpha0=0.001;
% hyperparameter: delta0
delta0=0.001;

% just for the code to run (do not touch)
ar=0;
PriorExcel=0;
priorsexogenous=1;
lambda4=100;
end



% Model options
% activate impulse response functions (1=yes, 0=no)
IRF=1;
% activate unconditional forecasts (1=yes, 0=no)
F=1;
% activate forecast error variance decomposition (1=yes, 0=no)
FEVD=1;
% activate historical decomposition (1=yes, 0=no)
HD=0;
% activate conditional forecasts (1=yes, 0=no)
CF=1;
% structural identification (1=none, 2=Choleski, 3=triangular factorisation, 4=sign restrictions)
IRFt=4;
% IRFt options
    % strctident settings for OLS model
    if VARtype==1
        if IRFt==4
        strctident.MM=0; % option for Median model (0=no (standard), 1=yes)
        % Correlation restriction options:
        strctident.CorrelShock=''; % exact labelname of the shock defined in one of the "...res values" excel sheets, otherwise if the shock is not identified yet name it 'CorrelShock'
        strctident.CorrelInstrument=''; % provide the IV variable in excel sheet "IV"        
        elseif IRFt==5
        % IV options:
        strctident.Instrument='MHF';% specify Instrument to identfy Shock in excel sheet "IV" 
        strctident.startdateIV='1992m2';
        strctident.enddateIV='2003m12';           
        strctident.bootstraptype=1; %1=wild bootstrap Mertens&Ravn(2013), 2=moving block bootstrap Jentsch&Lunsford(2018)
        elseif IRFt==6
        strctident.MM=0; % option for Median model (0=no (standard), 1=yes)
        % IV options:
        strctident.Instrument='MHF';% specify Instrument to identfy Shock in excel sheet "IV" 
        strctident.startdateIV='1992m2';
        strctident.enddateIV='2003m12';           
        strctident.bootstraptype=1; %1=wild bootstrap Mertens&Ravn(2013), 2=moving block bootstrap Jentsch&Lunsford(2018)
        strctident.TakeOLS=0; %only for IRFt6, OLS D and median irf_estimates
        % Correlation restriction options:
        strctident.CorrelShock='CorrelShock'; % exact labelname of the shock defined in one of the "...res values" excel sheets, otherwise if the shock is not identified yet name it 'CorrelShock'
        strctident.CorrelInstrument='MHF'; % provide the IV variable in excel sheet "IV" 
        end
    % strctident settings for Bayesian model
    elseif VARtype==2
        if IRFt==4
        strctident.MM=0; % option for Median model (0=no (standard), 1=yes)
        % Correlation restriction options:
        strctident.CorrelShock=''; % exact labelname of the shock defined in one of the "...res values" excel sheets, otherwise if the shock is not identified yet name it 'CorrelShock'
        strctident.CorrelInstrument=''; % provide the IV variable in excel sheet "IV"            
        elseif IRFt==5
        strctident.MM=0; % option for Median model (0=no (standard), 1=yes)
        % IV options:
        strctident.Instrument='MHF'; % specify Instrument to identfy Shock in excel sheet "IV" 
        strctident.startdateIV='1992m2';
        strctident.enddateIV='2003m12';
        strctident.Thin=10;
        strctident.prior_type_reduced_form=1; %1=flat (standard), 2=normal wishart , related to the IV routine
        strctident.Switchprobability=0; % (=0 standard) related to the IV routine, governs the believe of the researcher if the posterior distribution of Sigma|Y as specified by the standard inverse Wishart distribution, is a good proposal distribution for Sigma|Y, IV. If gamma = 1, beta and sigma are drawn from multivariate normal and inverse wishart. If not Sigma may be drawn around its previous value if randnumber < gamma
        strctident.prior_type_proxy=1; %1=inverse gamma (standard) 2=high relevance , related to the IV routine, priortype for the proxy equation (relevance of the proxy)
        elseif IRFt==6
        strctident.MM=0; % option for Median model (0=no (standard), 1=yes)
        % IV options:
        strctident.Instrument='MHF'; % specify Instrument to identfy Shock in excel sheet "IV" 
        strctident.startdateIV='1992m2';
        strctident.enddateIV='2003m12';
        strctident.Thin=10;
        strctident.prior_type_reduced_form=1; %1=flat (standard), 2=normal wishart , related to the IV routine
        strctident.Switchprobability=0; % (=0 standard) related to the IV routine, governs the believe of the researcher if the posterior distribution of Sigma|Y as specified by the standard inverse Wishart distribution, is a good proposal distribution for Sigma|Y, IV. If gamma = 1, beta and sigma are drawn from multivariate normal and inverse wishart. If not Sigma may be drawn around its previous value if randnumber < gamma
        strctident.prior_type_proxy=1; %1=inverse gamma (standard) 2=high relevance , related to the IV routine, priortype for the proxy equation (relevance of the proxy)
        % Correlation restriction options:
        strctident.CorrelShock='CorrelShock'; % exact labelname of the shock defined in one of the "...res values" excel sheets, otherwise if the shock is not identified yet name it 'correl.shock'
        strctident.CorrelInstrument='MHF'; % provide the IV variable in excel sheet "IV"
        end
    end
    
% activate forecast evaluation (1=yes, 0=no)
Feval=1;
% type of conditional forecasts 
% 1=standard (all shocks), 2=standard (shock-specific)
% 3=tilting (median), 4=tilting (interval)
CFt=1;
% number of periods for impulse response functions
IRFperiods=20;
% start date for forecasts (has to be an in-sample date; otherwise, ignore and set Fendsmpl=1)
Fstartdate='2014q1';
% end date for forecasts
Fenddate='2016q4';
% start forecasts immediately after the final sample period (1=yes, 0=no)
% has to be set to 1 if start date for forecasts is not in-sample
Fendsmpl=0;
% step ahead evaluation
hstep=1;
% window_size for iterative forecasting 0 if no iterative forecasting
window_size=0; 
% evaluation_size as percent of window_size                                      <                                                                                    -
evaluation_size=0.5;                          
% confidence/credibility level for VAR coefficients
cband=0.95;
% confidence/credibility level for impusle response functions
IRFband=0.95;
% confidence/credibility level for forecasts
Fband=0.95;
% confidence/credibility level for forecast error variance decomposition
FEVDband=0.95;
% confidence/credibility level for historical decomposition
HDband=0.95;