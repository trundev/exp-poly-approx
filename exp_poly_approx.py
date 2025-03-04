"""Interpolation/approximation via exponential polynomials

References:
- YouTube video by Mathologer: https://www.youtube.com/watch?v=NO1_-qptr6c
- Exponential polynomials: https://en.wikipedia.org/wiki/Exponential_polynomial
"""
import numpy as np
import sympy
import sympy.abc
import sympy.core.evalf
from typing import Iterator
import numpy.typing as npt


DataArray = npt.NDArray
IndexArray = npt.NDArray[np.uint8]
PolyArray = npt.NDArray[np.object_]

# Vectorize some useful `sympy` functions
sympy_im = np.vectorize(sympy.im, otypes=[object])
sympy_log = np.vectorize(sympy.log, otypes=[object])
sympy_evalf = np.vectorize(sympy.core.evalf.EvalfMixin.evalf,
                           otypes=[complex], excluded=dict(chop=True))
sympy_latex = np.vectorize(sympy.latex, otypes=[str])

def poly_roots(poly_arr: PolyArray, symbol: sympy.Symbol) -> list[DataArray]:
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
        root_arr = np.fromiter(sympy.roots(poly, symbol).items(),
                               dtype=[('root', object), ('repeat', int)])
        res.append(root_arr)
    return res

def number_wall_cross(cur_row: DataArray, prev_row: DataArray) -> DataArray:
    """Calculate next number-wall row using "cross" rule

    Number-wall "cross" rule:
    - :math:`\frac{D_{-1,0}^2 - D_{-1,-1} D_{-1,+1}}{D_{-2,0}}`
    """
    next_row = cur_row[1:-1] * cur_row[1:-1]
    next_row -= cur_row[:-2] * cur_row[2:]
    return next_row / prev_row

def number_wall_gen(start_row: DataArray) -> Iterator[DataArray]:
    """Number-wall "crosses" generator"""
    cur_row = start_row
    yield cur_row
    # The 'previous' row: all ones
    prev_row = sympy.S.One
    # Build subsequent rows, by apply 'number-wall crosses' rule
    # Finish when all polynomials collapse or no data for the 'crosses'
    while cur_row.shape[0] > 2:
        next_row = number_wall_cross(cur_row, prev_row)
        # Return non-simplified form, to be displayed (may simplify to zero)
        yield next_row
        # Simplify for zero-test (and for next iteration)
        next_row = np.asarray(sympy.simplify(next_row))
        if not next_row.any():
            # All polynomials collapsed to zeros, stop the generator
            return
        # Swap to next row
        prev_row = cur_row[2:-2]
        cur_row = next_row
    # Polynomials may be independent (need more data)
    yield np.empty_like(cur_row, shape=0)

def poly_number_wall_gen(data: DataArray, symbol: sympy.Symbol) -> Iterator[PolyArray]:
    """Number-wall of polynomials generator

    Start row of the number-wall is array of 1-st degree polynomials:
    - :math:`a_1 - a_0 x`
    - :math:`a_2 - a_1 x`
    - ...
    """
    yield from number_wall_gen(data[1:] - data[:-1]*symbol)

def char_poly_number_wall(data: DataArray, symbol: sympy.Symbol) -> PolyArray:
    """Characteristic polynomial generator using number-wall approach

    Run number-wall of polynomials till it collapses, take the one before the last
    """
    poly_queue = (np.empty_like(data, shape=0),)
    for poly_arr in poly_number_wall_gen(data, symbol):
        # Keep queue two elements only
        poly_queue = poly_queue[-1:] + (poly_arr,)
    # Note: poly_number_wall_gen() return non-simplified form
    return np.asarray(sympy.simplify(poly_queue[0]))

def select_exponent_bases(poly_arr: PolyArray, symbol: sympy.Symbol) -> tuple[DataArray, IndexArray]:
    """Calculate bases of the exponents, identify repeated ones

    The exponent bases are the roots of the characteristic polynomial(s).
    Repeated roots must be handled separately, as they are treated as single
    exponent multiplied by a polynomial, instead of just a constant.
    """
    root_list = poly_roots(poly_arr, symbol)
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
              f' {np.abs(sympy_evalf(mean_root_arr - root_arr)).max()}', file=sys.stderr)
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

def build_exp_poly(coefs: DataArray, powers: IndexArray,
                   bases: DataArray, base_repeats: IndexArray, *,
                   symbol: sympy.Symbol=sympy.abc.x) -> sympy.Expr:
    """Construct exponential polynomial"""
    assert coefs.shape == powers.shape, 'Polynomial coefs do not match powers'
    assert base_repeats.sum(0) == powers.shape[0], 'Exponential repeats do not match coefs'
    # The simplest way is to just do:
    # ```
    # bases = np.repeat(bases, base_repeats)
    # exp_poly = (coefs * bases ** symbol * symbol ** powers).sum(0)`
    # ```
    # However, this will distribute each exponent over its polynomial components,
    # which loses the structure of exponential-polynomial.
    #
    # Combine polynomials from their components based on corresponding exponent
    polys = np.full_like(bases, sympy.S.Zero)
    base_idx = np.repeat(np.arange(base_repeats.shape[0]), base_repeats)
    np.add.at(polys, base_idx, coefs * symbol ** powers)
    # Then multiply each one by its exponent and sum, if empty result is still sympy.Expr
    return (polys * bases ** symbol).sum(0, initial=sympy.S.Zero)

def interpolate_number_wall(data: DataArray, *, simplify: bool=False) -> sympy.Expr:
    """Generate exponential polynomial to interpolate given data

    This uses number-wall of polynomials approach:
    - Build characteristic polynomial from number-wall crosses
    - Use its roots as the exponent bases
    - Solve for exponent coefficients (which are polynomials)
    """
    # Find characteristic polynomial, its roots are the exponent bases
    symbol = sympy.abc.x
    char_poly_arr = char_poly_number_wall(data, symbol)
    print(f'  Characteristic polynomial(s): {char_poly_arr}')
    bases, repeats = select_exponent_bases(char_poly_arr, symbol)
    print(f'  Exponent bases (averaged): {bases}, multiplicity {repeats}')
    if sympy_im(bases).any():
        angles = sympy_im(sympy_log(bases))
        print(f'    period: {2*sympy.pi/angles} samples')
    coefs, powers = solve_coefficients(data, bases, repeats)
    print(f'  Polynomial coefficients: {coefs}')
    exp_poly = build_exp_poly(coefs, powers, bases, repeats)
    return sympy.simplify(exp_poly) if simplify else exp_poly

def main(argv):
    """Standalone execution"""
    data = sympy.sympify(argv)
    data = np.squeeze(np.asarray(data))
    print(f'Input data: {data}')
    exp_poly = interpolate_number_wall(data)
    print(f'  Decomposed function: {sympy.latex(exp_poly)}')
    print(f'    Simplified: {sympy.latex(sympy.simplify(exp_poly))}')
    reval = sympy.lambdify(sympy.abc.x, exp_poly, 'numpy')(np.arange(data.size + 3))
    print(f'  Reevaluate: {reval}')

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
