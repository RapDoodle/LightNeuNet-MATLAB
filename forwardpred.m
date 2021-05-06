function forwardpred(model, X, y)
    probs = model.predict(X);
    [~, y_out] = max(probs, [], 1);
    pred = bsxfun(@eq, y_out, (1:10)');
    correct = find(all(pred == y));
    accuracy = length(correct) / size(y, 2);
    model.fitness = accuracy * 100;
    clear X;
    clear y;
end

