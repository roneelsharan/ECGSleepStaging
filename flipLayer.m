%% Used for bi-directional GRU
classdef flipLayer < nnet.layer.Layer
    methods
        function layer = flipLayer(name)
            layer.Name = name;
        end
        function Y = predict(~, X)
            Y = flip(X,3);
        end
    end
end