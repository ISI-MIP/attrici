import theano.tensor as tt

def det_dot(a, b):
    """
    The theano dot product and NUTS sampler don't work with large matrices?

    :param a: (np matrix)
    :param b: (theano vector)
    """
    return (a * b[None, :]).sum(axis=-1)


def full(gmt, p_intercept, p_slope, p_yearly, p_trend, xf_yearly, xf_trend):

    return p_intercept / (
        1
        + tt.exp(
            -1
            * (
                p_slope * gmt
                + det_dot(xf_yearly, p_yearly)
                + gmt * det_dot(xf_trend, p_trend)
            )
        )
    )

def longterm_yearlycycle(gmt, p_intercept, p_slope, p_yearly, xf_yearly):

    return p_intercept / (
        1 + tt.exp(-1 * (p_slope * gmt + det_dot(xf_yearly, p_yearly)))
    )

def yearlycycle(p_intercept, p_yearly, xf_yearly):

    return p_intercept / (1 + tt.exp(-1 * det_dot(xf_yearly, p_yearly)))

