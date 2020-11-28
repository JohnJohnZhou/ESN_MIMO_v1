function C = UnitQamConstellation(Bi)


% Check for the trivial cases of less than 1 bit (erroneous input) or 1 bit (the
% BPSK case).  All others should be OK with our main section of code.
if(Bi < 1)
    C = [];
    return;
elseif(Bi == 1)
    C = [-1;+1];
    return;
end


% When Bi is even, then M has the following properties:
%   * it has an integer square root
%   * the square root is an even number
% When Bi is odd, we want to round its square root up to the next even number,
% in order to find the smallest even-sided square that will hold our points.
EvenSquareRoot = ceil(sqrt(2^Bi)/2)*2;


% We need a PAM-type alphabet based on this even square root
PamM = EvenSquareRoot;

% Now, make the square QAM constellation using the basic PAM order:
%   * Start with the basic M-ary PAM constellation
%   * Make an M-by-M matrix where each row is the basic M-ary PAM constellation
%   * Make a copy of the M-by-M matrix, and then transpose the copy
%   * Multiply the first matrix by 1, and the second matrix by j, then add
PamConstellation = -(PamM-1):2:+(PamM-1);
SquareMatrix     = ones(PamM,1)*PamConstellation;
C                = SquareMatrix + j*SquareMatrix';
C=C(:);

% If Bi is even, then we're done.  If Bi is odd, then we are dealing with a
% "cross" constellation, and we have to keep only the M points that are closest
% to the origin
if(mod(Bi,2) == 1)
    % There will be a few "ties" when we sort by minimum distance, so MATLAB
    % will use some sort of "tiebreaker."  Therefore, we will just grab the
    % constellation points in the first quadrant.  Then, we will replicate the
    % first quadrant 4 times.  This way we end up with a constellation that is
    % symmetric looking.
    FirstQuadrant = find( (real(C) > 0) & (imag(C) > 0) );
    C = C(FirstQuadrant);
    d = abs(C);
    [dSort,ISort] = sort(d);
    C = C(ISort(1:2^Bi/4));
    
    % Replicate the first quadrant 4 times
    C = [real(C) + j*imag(C);
         real(C) - j*imag(C);
        -real(C) + j*imag(C);
        -real(C) - j*imag(C);];
end


% TODO: Normalize the constellation so that it has an average energy of unity.
C = C/sqrt(mean(abs(C).^2));
