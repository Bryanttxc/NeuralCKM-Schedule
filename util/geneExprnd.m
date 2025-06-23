% created by Bryanttxc, FUNLab

function result = geneExprnd(lambda, d_1)
    result = zeros(d_1, 1);
    for p = 1:d_1
        temp = exprnd(1/lambda);
        while temp > 1
            temp = exprnd(1/lambda);
        end
        result(p) = temp;
    end
end

