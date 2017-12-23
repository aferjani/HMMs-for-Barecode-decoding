function patterns = getPatternsUPCA()

    patterns = cell(25,1);
    ix = 1;
    patterns{ix} = [ 0  0  0  0  0  0  0];    %1: space begin
    ix = ix+1;
    patterns{ix} = [ 0  0  0  0  0  0  0];    %2: space end
    ix = ix+1;
    patterns{ix} = [ 1  0  1];                %3: guard begin
    ix = ix+1;
    patterns{ix} = [ 1  0  1];                %4: guard end
    ix = ix+1;
    patterns{ix} = [ 0  1  0  1  0];          %5: guard middle
    ix = ix+1;
    patterns{ix} = [ 0  0  0  1  1  0  1];    %6: 0 left
    ix = ix+1;
    patterns{ix} = [ 0  0  1  1  0  0  1];    %7: 1 left
    ix = ix+1;
    patterns{ix} = [ 0  0  1  0  0  1  1];    %8: 2 left
    ix = ix+1;
    patterns{ix} = [ 0  1  1  1  1  0  1];    %9: 3 left
    ix = ix+1;
    patterns{ix} = [ 0  1  0  0  0  1  1];    %10: 4 left
    ix = ix+1;
    patterns{ix} = [ 0  1  1  0  0  0  1];    %11: 5 left
    ix = ix+1;
    patterns{ix} = [ 0  1  0  1  1  1  1];    %12: 6 left
    ix = ix+1;
    patterns{ix} = [ 0  1  1  1  0  1  1];    %13: 7 left
    ix = ix+1;
    patterns{ix} = [ 0  1  1  0  1  1  1];    %14: 8 left
    ix = ix+1;
    patterns{ix} = [ 0  0  0  1  0  1  1];    %15: 9 left
    ix = ix+1;
    
    for i = 6:15
        patterns{ix} =  1-patterns{i}; %16-15: right parts
        ix = ix+1;
    end
end