import marimo

__generated_with = "0.11.20"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""# SVD for Image Compression""")
    return


@app.cell
def _(mo, np, plt):
    iris = plt.imread('public/svd_image_compression_iris.png')
    iris = np.dot(iris[..., :3], [0.2989, 0.5870, 0.1140])

    (U, S, VT) = np.linalg.svd(iris, full_matrices=False)
    S_vals = S.copy()
    S = np.diag(S)

    r_min, r_max = 1, len(S_vals)
    r_slider = mo.ui.slider(r_min, r_max, value=r_min, show_value=True, full_width=True, label='Rank')

    return S, S_vals, U, VT, iris, r_max, r_min, r_slider


@app.cell
def _(S, S_vals, U, VT, iris, mo, np, plt, r_slider):
    r = r_slider.value
    w, h = 350, 350
    hori_images = mo.hstack([
        mo.image(iris, width=w, height=h),
        mo.image(U[:,:r] @ S[:r, :r] @ VT[:r, :], width=w, height=h)
    ])

    fig_S_vals = plt.figure()
    plt.semilogy(S_vals)
    plt.axvline(x=r, color='r')
    plt.title('Singular Values Sorted by Magnitude')
    plt.ylabel(r'Singular value, $\sigma$')
    plt.xlabel(r'Rank, $r$')

    fig_S_cumsum = plt.figure()
    plt.plot(np.cumsum(S_vals) / np.sum(S_vals))
    plt.axvline(x=r, color='r')
    plt.title('Cumulative Sum of Singular Values')
    plt.ylabel('Cumulative sum')
    plt.xlabel(r'Rank, $r$')

    hori_plots = mo.hstack([
        fig_S_vals,
        fig_S_cumsum
    ])

    comp_ratio_text = \
    r"""
    $$\text{Compression Ratio}, C = \frac{mr + r + nr}{mn}  = \frac{r(m + 1 + n)}{mn}$$.
    """
    accord_equations = mo.accordion(
        {
            "Equations"   : mo.md(comp_ratio_text),
        }
    )
    # vert_definitions = mo.vstack([
    #     mo.md(comp_ratio_text),
    #     mo.md(r"$m$ = Number of rows of image"),
    #     mo.md(r"$n$ = Number of columns of image"),
    #     mo.md(r"$r$ = Rank of approximation"),
    # ])

    m, n = iris.shape
    C = (r * (m + 1 + n)) / (m * n)
    hori_compression = mo.hstack([
        mo.md(f"C = {C:.3f}"),
        mo.md(f"Stores {C*100:.3f}% of the data."),
        mo.md(f"Cumulative sum of singular values = {np.sum(S_vals[:r] / np.sum(S_vals)):.3f}")
    ])

    mo.vstack([
        r_slider,
        accord_equations,
        hori_compression,
        hori_images,
        hori_plots
    ])

    return (
        C,
        accord_equations,
        comp_ratio_text,
        fig_S_cumsum,
        fig_S_vals,
        h,
        hori_compression,
        hori_images,
        hori_plots,
        m,
        n,
        r,
        w,
    )


if __name__ == "__main__":
    app.run()
