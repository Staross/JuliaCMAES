function dCorr(x,y)

    x = vec(x)
    y = vec(y)

    nx = length(x)
    ny = length(y)

    if(nx != ny)
        error("x and y must have the same length")
    end

    function pdist(z::Vector)
        n=length(z)
        dz = zeros(n,n)
        for i in 1:n
            for j in i+1:n
                d = abs( z[i]-z[j] )
                dz[i,j] = d
                dz[j,i] = d
            end
        end
        return dz
    end

    dx = pdist(x)
    dy = pdist(y)

    mi = repmat(mean(dx,1),nx,1)
    mj = repmat(mean(dx,2),1,nx)

    m = mean(dx)

    X = dx-mi-mj+m

    mi = repmat(mean(dy,1),ny,1)
    mj = repmat(mean(dy,2),1,ny)

    m = mean(dy)

    Y = dy-mi-mj+m

    V = Y.*X
    V = sqrt( mean(V) )

    Vx = X.*X
    Vx = sqrt( mean(Vx) )

    Vy = Y.*Y
    Vy = sqrt( mean(Vy) )

    R =  V / sqrt(Vx*Vy)

    r=500
    #bootstrap
    M=0
    for i=1:r

        ind = randperm(length(x))

        Xp=X[ind,:]
        Xp=Xp[:,ind]

        Vp = Y.*Xp
        Vp = mean(Vp)
        Vp = max(Vp,0)
        Vp = sqrt( Vp )

        if(Vp>=V)
            M=M+1
        end

    end

    pval = (M+1) / (r+1)

    return R, pval

end
