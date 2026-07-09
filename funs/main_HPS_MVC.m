function [F,obj] = main_HPS_MVC(X,beta,mu,k,t,M,c)

V=length(X);
num=size(X{1},2);
L=length(M);

Z=cell(V,L);
A=cell(V,L);

XMK=cell(V,L);
alpha=ones(V,L)./(V*L);
alpha_h=ones(V,L);

F=zeros(num,c);
F(1:c,:) = eye(c);
len_ml=zeros(L,1);

for l=1:L
    len_ml(l)=size(M{l},2);
    temp_Z=zeros(k,len_ml(l)); % m  * n
    temp_Z(:,1:k) = eye(k);
    temp_K=sum(M{l},1);

    for v = 1 :V
        Z{v,l}=sparse(temp_Z);

        XM=fast_cal(X{v},M{l});
        XMK{v,l}=XM.*(1./temp_K);
    end
end



for iter=1:30
    iter

    % update A^v
    for v = 1 :V
        for l=1:L
            [Uu1,~,Vv1] = svd(XMK{v,l}*Z{v,l}','econ');
            A{v,l} = Uu1*Vv1';
        end
    end

    % update Z^v
    for l=1:L
        temp2=fast_cal(F',M{l});
        Q=temp2'*temp2;
        for v=1:V
            W=A{v,l}'*XMK{v,l};
            temp3=2*beta*alpha(v,l)*alpha_h(v,l)*Z{v,l};


            Zv=Z{v,l};
            for i=1:len_ml(l)
                q_i=Q(:,i);
                q_i(i)=0;
                temp1=W(:,i)+temp3*q_i;

                tempZ = EProjSimplex_new(0.5*temp1/(mu-beta*alpha(v,l)*alpha_h(v,l)*Q(i,i)));
                Zv(:,i)=tempZ;
            end
            Z{v,l}=sparse(Zv);
        end
    end
    clear temp1 temp2 temp3
 
    % update F
    P=[];
    for v=1:V
        for l=1:L
            P=[P,M{l}*Z{v,l}'*sqrt(alpha(v,l)*alpha_h(v,l))];
        end
    end
    [nn, ~, ~] = svd(full(P), 'econ');
    F=nn(:,1:c);
    clear P


    % update alpha
    FTMZT=cell(V,L);
    h=zeros(V,L);
    for v=1:V
        for l=1:L
            FTM=fast_cal(F',M{l});
            FTMZT{v,l}=FTM*Z{v,l}';
            h(v,l)=sum(FTMZT{v,l}.^2,'all');
        end
    end
    h_vector=reshape(h,[V*L 1]);
    h_vector_s=sort(h_vector);
    h_vector(h_vector<=h_vector_s(t))=1;
    h_vector(h_vector~=1)=0;
    alpha_h=reshape(h_vector,[V L]);
    alpha_h_h=alpha_h.*h;
    alpha=alpha_h_h./sqrt(sum(alpha_h_h.^2,'all'));

    % calculate obj
    obj1=0;
    obj2=0;
    obj3=0;
    for v=1:V
        for l=1:L
            temp1=A{v,l}*Z{v,l};
            temp2=FTMZT{v,l};

            obj1=obj1+sum(XMK{v,l}.*temp1,'all');
            obj2=obj2+sum(temp2.^2,'all')*alpha(v,l)*alpha_h(v,l);
            obj3=obj3+sum(Z{v,l}.^2,'all');
        end
    end

    obj(iter)=obj1+beta*obj2-mu*obj3;
    if iter>3&&abs((obj(iter)-obj(iter-1))/obj(iter-1))<10^(-5)
        break
    end
end

% figure(1)
% plot(1:length(obj),obj)

end