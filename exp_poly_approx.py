"""Approximation via exponential polynomials

References:
- YouTube video by Mathologer: https://www.youtube.com/watch?v=NO1_-qptr6c
- Exponential polynomials: https://en.wikipedia.org/wiki/Exponential_polynomial
"""
import numpy as np
import sympy
import sympy.abc
import numpy.typing as npt


DataArray = npt.NDArray
IndexArray = npt.NDArray[np.uint8]
PolyArray = npt.NDArray[np.object_]

def poly_roots(poly_arr: PolyArray) -> list[DataArray]:
    """Wrapper for `sympy.polys.polyroots.roots()`

    Returns:
    List of roots for each polynomial.
    Roots are represented as array of structured data type:
    - 'root': sympy.Number - value of polynomial root
    - 'repeat': integer - multiplicity of polynomial root
    """
    res = []
    for poly in poly_arr:
        # Convert to array of structured data type
        root_arr = np.fromiter(sympy.roots(poly).items(),
                               dtype=[('root', object), ('repeat', int)])
        res.append(root_arr)
    return res

def build_char_poly(data: npt.NDArray, symbol: sympy.Symbol) -> PolyArray:
    """Characteristic polynomial"""
    #
    # Starting row of the number-wall
    # Array of 1-st degree polynomials:
    # - :math:`a_1 - a_0 x`
    # - :math:`a_2 - a_1 x`
    # - ...
    #
    cur_row = data[1:] - data[:-1]*symbol
    # The 'previous' one: all ones
    prev_row = sympy.S.One
    # Build subsequent rows, by apply 'number-wall crosses' rule
    # Finish when all polynomials collapse or no data for the 'crosses'
    while cur_row.shape[0] > 2:
        #
        # Number-wall crosses rule
        # - :math:`\frac{D_{-1,0}^2 - D_{-1,-1} D_{-1,+1}}{D_{-2,0}}`
        #
        next_row = cur_row[1:-1] * cur_row[1:-1]
        next_row -= cur_row[:-2] * cur_row[2:]
        next_row /= prev_row
        next_row = sympy.simplify(next_row)
        # Convert back to numpy array as it is easier to work
        next_row = np.asarray(next_row)
        # Check if it is all zeros
        if not next_row.any():
            # All polynomials are collapsed, return the last one
            return cur_row
        # Swap to next row
        prev_row = cur_row[2:-2]
        cur_row = next_row
    # Polynomials may be independent (need more data)
    return cur_row

def calc_exponent_bases(data: npt.NDArray) -> tuple[DataArray, IndexArray]:
    """Calculate bases of the exponents, identify repeated ones

    The exponent bases are the roots of the characteristic polynomial(s).
    Repeated roots must be handled separately, as they are treated as single
    exponent multiplied by a polynomial, instead of just a constant.
    """
    poly_arr = build_char_poly(data, sympy.abc.x)
    root_list = poly_roots(poly_arr)
    if not root_list:
        return np.empty(0), np.empty(0, dtype=int)
    #TODO: Expect equal number of roots
    root_arr = np.asarray(root_list)
    # Split to polynomial roots and root-repeats
    root_repeat: IndexArray = root_arr['repeat']
    root_arr: DataArray = root_arr['root']
    np.testing.assert_equal(root_repeat[1:], root_repeat[:-1],
                            err_msg='Expected identical root repetition')
    # Average roots from all returned polynomials (expect to be the same)
    mean_root_arr = root_arr.mean(0)
    if (mean_root_arr - root_arr).any():
        print('Warning: Characteristic polynomial root deviation: '
              f' {(mean_root_arr - root_arr).max()}')
    return mean_root_arr, root_repeat[0]

def solve_coefficients(data: DataArray, exp_bases: DataArray, base_repeats:IndexArray,
                       x0=0) -> tuple[DataArray, IndexArray]:
    """Find the coefficients of polynomial multiplying each exponent"""
    exp_bases = np.repeat(exp_bases, base_repeats)
    # Select powers of the polynomial:
    # Repeated exponent bases share the same polynomial,
    # so base-repetition increases the x-power (polynomial degree)
    mask = np.repeat(np.arange(base_repeats.shape[0]), base_repeats)
    mask = mask[1:] == mask[:-1]
    x_powers = np.zeros_like(base_repeats, shape=base_repeats.sum(0))
    idx = 1
    while mask.any():
        x_powers[idx:] += mask
        mask = mask[1:] & mask[:-1]
        idx += 1
    # Range of the polynomial argument (x)
    x_range = x0 + np.arange(exp_bases.shape[-1], dtype=exp_bases.dtype)[...,np.newaxis]
    # Spread polynomial: x^{n}
    equation_mtx = x_range ** x_powers
    # Spread exponent: b^{x}
    equation_mtx *= exp_bases ** x_range
    # Solve the system of equations
    coefs = sympy.Matrix(equation_mtx) ** -1 * sympy.Matrix(data[:exp_bases.size])
    return np.asarray(coefs)[:,0], x_powers

def build_exp_poly(coefs: DataArray, powers: IndexArray, bases: DataArray, *,
             symbol: sympy.Symbol=sympy.abc.x) -> sympy.Basic:
    """Construct exponential polynomial"""
    exp_poly = (coefs * bases ** symbol * symbol ** powers).sum(0)
    return sympy.simplify(exp_poly)

def approximate_data(data: npt.NDArray) -> sympy.Basic:
    """Generate exponential polynomial to approximate given data"""
    bases, repeats = calc_exponent_bases(data)
    coefs, powers = solve_coefficients(data, bases, repeats)
    return build_exp_poly(coefs, powers, np.repeat(bases, repeats))

def main(argv):
    """Standalone execution"""
    dtype = int
    data = np.fromiter((dtype(v) for v in argv), dtype=dtype)
    print(f'Input data: {data}')
    bases, repeats = calc_exponent_bases(data)
    print(f'  Exponent bases (averaged): {bases}, multiplicity {repeats}')
    sympy_im = np.vectorize(sympy.im, otypes=[object])
    if sympy_im(bases).any():
        angles = np.vectorize(sympy.log, otypes=[object])(bases)
        angles = sympy_im(angles)
        print(f'    period: {2*sympy.pi/angles} samples')
    coefs, powers = solve_coefficients(data, bases, repeats)
    print(f'  Polynomial coefficients: {coefs}')
    exp_poly = build_exp_poly(coefs, powers, np.repeat(bases, repeats))
    print(f'  Decomposed function: {sympy.latex(exp_poly)}')

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
