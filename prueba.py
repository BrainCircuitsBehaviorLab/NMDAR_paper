# /// script
# dependencies = ["marimo"]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    return np, plt


@app.cell
def _(mo):
    a = mo.ui.number(start=-20, stop=20, label="a")
    b = mo.ui.number(start=-20, stop=20, label="b")
    c = mo.ui.number(start=-20, stop=20, label="c")
    mo.hstack([a,b,c])
    return a, b, c


@app.cell
def _(a, b, c, np, plt):
    # Define x range
    x = np.linspace(-10, 10, 400)

    # Define function
    y = a.value*x**2 + b.value * x + c.value

    # Plot
    fig = plt.figure()
    plt.plot(x, y)
    def y(x):
        return a.value*x**2 + b.value * x + c.value

    # Highlight vertex
    vx = -b.value/(2*a.value) 
    plt.scatter(vx, y(vx))

    # Axis lines
    plt.axhline(0)
    plt.axvline(0)

    plt.title(f"Parabola: y = {a.value}x^2 + {b.value}x + {c.value}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.xlim(-10,10)
    plt.ylim(-10,10)

    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
