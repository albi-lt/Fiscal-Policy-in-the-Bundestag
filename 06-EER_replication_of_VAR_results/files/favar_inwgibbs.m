function [beta_gibbs,sigma_gibbs,favar,It,Bu]=favar_inwgibbs(It,Bu,B,EPS,n,T,q,lags,data_endo,data_exo,const,favar,ar,arvar,lambda1,lambda2,lambda3,lambda4,lambda5,m,p,k,prior,bex,blockexo,priorexo,Y,X,y,endo)

%% references: Bernanke, Boivin, Eliasz (2005), Koop & Korobilis

%% preliminary tasks
% initialise variables
nfactorvar=favar.nfactorvar;
numpc=favar.numpc;
favarX=favar.X(:,favar.plotX_index);
favarplotX_index=favar.plotX_index;
onestep=favar.onestep;
XY=favar.XY;

% initial conditions XZ0~N(XZ0mean,XZ0var)
favar.XZ0mean=zeros(n*lags,1);
favar.XZ0var=favar.L0*eye(n*lags); %BBE set-up

L=favar.L;
Sigma=nspd(favar.Sigma);
if onestep==1
    indexnM=favar.indexnM;
end
XZ0mean=favar.XZ0mean;
XZ0var=favar.XZ0var;
favar_X=favar.X;
% load priors
L0=favar.L0*eye(n);
a0=favar.a0;
b0=favar.b0;
sigmahat=(1/T)*(EPS'*EPS);

% preallocation
beta_gibbs=zeros(size(B(:),1),It-Bu);
sigma_gibbs=zeros(size(sigmahat(:),1),It-Bu);
X_gibbs=zeros(size(X(:),1),It-Bu);
Y_gibbs=zeros(size(Y(:),1),It-Bu);
FY_gibbs=zeros(size(data_endo(:),1),It-Bu);
L_gibbs=zeros(size(L(:),1),It-Bu);
R2_gibbs=zeros(size(favarX,2),It-Bu);

if onestep==0 %static factors in this case
    FY=data_endo;
    pbstring='two-step'; %string for the progress bar
elseif onestep==1
    pbstring='one-step'; %string for the progress bar
end

% state-space representation
if onestep==1
    B_ss=[B';eye(n*(lags-1)) zeros(n*(lags-1),n)];
    sigma_ss=[sigmahat zeros(n,n*(lags-1));zeros(n*(lags-1),n*lags)];
elseif onestep==0
    %set prior values
    [beta0,omega0,S0,alpha0]=inwprior(ar,arvar,lambda1,lambda2,lambda3,lambda4,lambda5,n,m,p,k,q,prior,bex,blockexo,priorexo);
    % invert omega0, as it will be used repeatedly
    invomega0=diag(1./diag(omega0));
    % set the value of alphahat, defined in (1.5.16)
    alphahat=T+alpha0;
end
% create a progress bar
hbar = parfor_progressbar(It,['Progress of the Gibbs sampler (',pbstring,').']);

%% start iterations
for ii=1:It
    if onestep==1
        % Sample latent factors using Carter and Kohn (1994)
        FY=favar_kfgibbsnv(XY,XZ0mean,XZ0var,L,Sigma,B_ss,sigma_ss,indexnM);
        % demean generated factors
        FY=favar_demean(FY);
        % Sample autoregressive coefficients B
        [B,~,~,X,~,Y,y]=olsvar(FY,data_exo,const,lags);
        [arvar]=arloop(FY,const,p,n);
        % set prior values, new with every iteration for onestep only
        [beta0,omega0,S0,alpha0]=inwprior(ar,arvar,lambda1,lambda2,lambda3,lambda4,lambda5,n,m,p,k,q,prior,bex,blockexo,priorexo);
        % invert omega0, as it will be used repeatedly
        invomega0=diag(1./diag(omega0));
        % set the value of alphahat, defined in (1.5.16)
        alphahat=T+alpha0;
    end
    
    % Step 3: at iteration ii, first draw sigma from IW, conditional on beta from previous iteration
    % obtain first Shat, defined in (1.5.15)
    Shat=(Y-X*B)'*(Y-X*B)+S0;
    % Correct potential asymmetries due to rounding errors from Matlab
    Shat=nspd(Shat);
    % next draw from IW(Shat,alphahat)
    sigma=iwdraw(Shat,alphahat);
    if onestep==1
        sigma_ss(1:n,1:n)=sigma;
    end
    
    % step 4: with sigma drawn, continue iteration ii by drawing beta from a multivariate Normal, conditional on sigma obtained in current iteration
    % first invert sigma
    C=trns(chol(nspd(sigma),'Lower'));
    invC=C\speye(n);
    invsigma=invC*invC';
    
    % then obtain the omegabar matrix
    invomegabar=invomega0+kron(invsigma,X'*X);
    C=chol(nspd(invomegabar));
    invC=C\speye(q);
    omegabar=invC*invC';
    
    % following, obtain betabar
    betabar=omegabar*(invomega0*beta0+kron(invsigma,X')*y);
    
    % draw beta from N(betabar,omegabar);
    stationary=0;
    while stationary==0
        % draw from N(betabar,omegabar);
        beta=betabar+chol(nspd(omegabar),'lower')*mvnrnd(zeros(q,1),eye(q))';
        [stationary]=checkstable(beta,n,lags,size(B,1)); %switches stationary to 0, if the draw is not stationary
    end
    
    % update matrix B with each draw
    B=reshape(beta,size(B));
    
    if onestep==1
        B_ss(1:n,:)=B';
        % Sample Sigma and L
        [Sigma,L]=favar_SigmaL(Sigma,L,nfactorvar,numpc,onestep,n,favar_X,FY,a0,b0,T,lags,L0);
    end
    
    %% record the values if the number of burn-in iterations is exceeded
    if ii>Bu
        % values of vector beta
        beta_gibbs(:,ii-Bu)=beta;
        % values of sigma (in vectorized form)
        sigma_gibbs(:,ii-Bu)=sigma(:);
        
        % save the factors and loadings
        X_gibbs(:,ii-Bu)=X(:);
        Y_gibbs(:,ii-Bu)=Y(:);
        FY_gibbs(:,ii-Bu)=FY(:);
        L_gibbs(:,ii-Bu)=L(:);
        
        % compute R2 (Coefficient of Determination) for plotX variables
        R2=favar_R2(favarX,FY,L,favarplotX_index);
        R2_gibbs(:,ii-Bu)=R2(:);
        
        % if current iteration is still a burn iteration, do not record the result
    else
    end
    
    % update progress by one iteration
    hbar.iterate(1);
    
    % go for next iteration
end

% in case we have thinning of the draws,
thin=abs(round(favar.thin)); % should be a positive integer
if thin~=1
    beta_gibbs=beta_gibbs(:,thin:thin:end);
    sigma_gibbs=sigma_gibbs(:,thin:thin:end);
    X_gibbs=X_gibbs(:,thin:thin:end);
    Y_gibbs=Y_gibbs(:,thin:thin:end);
    FY_gibbs=FY_gibbs(:,thin:thin:end);
    L_gibbs=L_gibbs(:,thin:thin:end);
    R2_gibbs=R2_gibbs(:,thin:thin:end);
    It=(1/thin)*It;
    Bu=(1/thin)*Bu;
end

% save in favar structure
favar.X_gibbs=X_gibbs;
favar.Y_gibbs=Y_gibbs;
favar.FY_gibbs=FY_gibbs;
favar.L_gibbs=L_gibbs;
favar.R2_gibbs=R2_gibbs;

% close progress bar
close(hbar);

