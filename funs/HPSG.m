function [M] = HPSG(X,c,Y)
V=length(X);
for v=1:V
    Center{v,1}=X{v}';
    X{v}=X{v}';
end

num_view=zeros(V,1);
for v=1:V
    [F_set, num_cluster]=FINCH(X{v},[],[]);
    num_view(v)=length(num_cluster);
end


Q=cell(V,min(num_view));
P1=cell(V,min(num_view));
ACC=zeros(min(num_view),1);
M=cell(min(num_view),1);
for l=1:min(num_view)
    adj=cell(V,1);
    for v=1:V
        [idx,~, center] = signal_FINCH(Center{v,l},[]);
        Q{v,l}=sparse(ind2vec(idx')');
        Center{v,l+1}=center;
        if l>1
            P1{v,l}=P1{v,l-1}*Q{v,l};
        elseif l==1
            P1{v,1}=Q{v,l};
        end

        [~,idx1]=max(P1{v,l},[],2);
        adj{v} = sparse(double(idx1 == idx1'));
    end

    adj_sum = adj{1};
    for i = 2:length(adj)
        adj_sum = adj_sum + adj{i};
    end

%     T = adj_sum > fix(V / 2) + 1;
    T = adj_sum >= 0.8*V;
    [~, y] = graphconncomp(T, 'Directed', false);
    F=sparse(ind2vec(y)');
    M{l}=F;
    clear adj
end

end