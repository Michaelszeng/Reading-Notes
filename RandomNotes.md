## Fourier Transform

Credits: [https://www.youtube.com/watch?v=spUNpyF58BY]

Purpose: Broadly, decomposes a wave function into a combination of sine waves. Technically, it takes an arbitrary wave function $g(t)$ and transforms it into a new function $\hat{g}(f)$ that contains spikes at the generating sine wave frequencies.

The new function $\hat{g}(f)$ takes frequency as an input (and outputs an complex number that is the "intensity" at each frequency):

$$ \hat{g}(f) =\int_{t_1}^{t_2} g(t) e^{-2 \pi i ft} dt $$

Intuitively, this is what the function is doing:
- First, it takes $e^{-2 \pi i ft}$, which corresponds to some point on the unit circle in the complex plane. Recall that $e^{it}$ = $\text{cos}(t) + i ~\text{sin}(t)$, so increasing $t$ from $0$ to $2\pi$ is like walking around the edge of a unit circle. Then, the input $f$ controls the rate we travel around the unit circle.
- Multiplying $e^{-2 \pi i ft}$ by $g(t)$ effectively applies $g(t)$ onto that unit circle; it's like wrapping the wave function on the circle.
- Taking the integral is effectively taking the "average" of all the points in the complex plane (the red dot in the bottom left graph). "Average" in quotations because it is more of a sum than an average (it's not divided by $(t_2 - t_1)$), but this just has the effect of scaling $\hat{g}(f)$ up for longer wave signals.
- Taking the real part of the output (i.e. the x-coordinate of the red dot), we find that it spikes around the frequencies of the sine waves constructing $g(t)$ (see the spikes on the top right graph).

<img src="ReadingNotesSupplements/Fourier.png" alt="" style="width:100%; margin-top: 10px"/>