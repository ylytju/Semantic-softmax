function W = SAE(Data_x,Data_y, lambda)
% Semantic Linear autoencoder
% Input:
%       Data_x:    d*N matrix
%       Data_y:    k*N matrix
%       lambda: regularisation parameter
%
% Output:
%       W: k*d projection matrix
%%
tau = 1e-4;
A_Mat = Data_y*Data_y'+tau*eye(size(Data_y,1));
B_Mat = lambda*Data_x*Data_x'+tau*eye(size(Data_x,1));
C_Mat = (1+lambda)*Data_y*Data_x';
W     = sylvester(A_Mat,B_Mat,C_Mat);
W     = normalization(W);
end
