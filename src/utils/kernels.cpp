#include <vector>
#include <complex>
#include <algorithm>
#include <thread>
#include <cmath>
#include <array>

using std::vector;
using std::complex;
using std::array;

template <typename T>
using Tensor3D = std::vector<std::vector<std::vector<T>>>;
template <typename T>
using Tensor4D = std::vector<std::vector<std::vector<std::vector<T>>>>;
template <typename T>
using Tensor5D = std::vector<std::vector<std::vector<std::vector<std::vector<T>>>>>;
template <typename T>
using Tensor6D = std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<T>>>>>>;

complex<float> combine_spectra (
    const int lcut,
    const int mcut,
    const int nspecies,
    const Tensor4D<complex<float>>& ISOAP,
    const vector<float>& divfac
) {
    complex<float> PS(0.0f);
    complex<float> PSS, PKM;
    for (int l = 0; l <= lcut; ++l) {
        PSS = complex<float>(0.0f);
        for (int im = 0; im <= 2 * l; ++im) {
            for (int ik = 0; ik <= 2 * l + 1; ++ik) {
                for (int ix = 0; ix < nspecies; ++ix) {
                    const auto dcix = std::conj(ISOAP[ix][l][im][ik]);
                    PKM = complex<float>(0.0f);

                    for (int iy = 0; iy < nspecies; ++iy) {
                        PKM += ISOAP[iy][l][im][ik];
                    }

                    PSS += PKM * dcix;
                }
            }
        }
        PS += PSS * divfac[l];
    }
    return PS;
}

vector<vector<complex<float>>> fill_spectra(
    const int lval,
    const int lcut,
    const int mcut,
    const int nspecies,
    const Tensor4D<complex<float>>& ISOAP,
    const Tensor6D<float>& CG2
) {
    vector<vector<complex<float>>> PS(
        2 * lval + 1, 
        vector<complex<float>>(2 * lval + 1, complex<float>(0.0f))
    );
    complex<float> PKM;
    for (int l1 = 0; l1 <= lcut; ++l1) {
        for (int l = 0; l <= lcut; ++l) {
            for (int im = 0; im <= 2 * l; ++im) {
                for (int ik = 0; ik <= 2 * l; ++ik) {
                    for (int ix = 0; ix < nspecies; ++ix) {
                        const auto dcix = std::conj(ISOAP[ix][l][im][ik]);

                        const auto iim_min = std::max(0, lval + im - l - l1);
                        const auto iim_max = std::min(2 * lval, lval + im - l + l1);
                        const auto iik_min = std::max(0, lval + ik - l - l1);
                        const auto iik_max = std::min(2 * lval, lval + ik - l + l1);

                        for (int iim = iim_min; iim <= iim_max; ++iim) {
                            for (int iik = iik_min; iik <= iik_max; ++iik) {
                                PKM = complex<float>(0.0f);

                                const int im_idx = im - l + l1 - iim + lval;
                                const int ik_idx = ik - l + l1 - iik + lval;

                                if (0 <= im_idx && im_idx < mcut && 0 <= ik_idx && ik_idx < mcut) {
                                    for (int iy = 0; iy < nspecies; ++iy) {
                                        PKM += ISOAP[iy][l1][im_idx][ik_idx];
                                    }

                                    PS[iim][iik] += PKM * CG2[l][l1][im][ik][iim][iik] * dcix;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return PS;
}

vector<complex<float>> spherical_in(int n_max, complex<float> z) {
    vector<complex<float>> result(n_max + 1);
    
    if (abs(z) < 1e-10) {
        // Handle small z case
        result[0] = complex<float>(1.0f, 0.0f);
        for (int n = 1; n <= n_max; ++n) {
            result[n] = complex<float>(0.0f, 0.0f);
        }
        return result;
    }
    
    // For small z, use series expansion
    if (abs(z) < 10.0f) {
        for (int n = 0; n <= n_max; ++n) {
            complex<float> sum(0.0f, 0.0f);
            complex<float> term(1.0f, 0.0f);
            complex<float> z_pow(1.0f, 0.0f);
            
            // Calculate z^n
            for (int k = 0; k < n; ++k) {
                z_pow *= z;
            }
            
            // Series expansion: i_n(z) = z^n / (2n+1)!! * sum_{k=0}^inf (z^2/2)^k / (k! * (2n+2k+1)!!)
            for (int k = 0; k < 50; ++k) {  // 50 terms should be enough for convergence
                if (k == 0) {
                    term = z_pow;
                    // Calculate (2n+1)!!
                    for (int j = 1; j <= n; ++j) {
                        term /= complex<float>(2.0f * j + 1.0f, 0.0f);
                    }
                } else {
                    term *= z * z / complex<float>(2.0f * k * (2.0f * n + 2.0f * k + 1.0f), 0.0f);
                }
                sum += term;
                
                // Check convergence
                if (abs(term) < 1e-15 * abs(sum)) break;
            }
            result[n] = sum;
        }
    } else {
        // For large z, use asymptotic expansion or recurrence relations
        // Asymptotic form: i_n(z) ≈ exp(z) / (sqrt(2*pi*z)) for large z
        complex<float> exp_z = exp(z);
        complex<float> sqrt_factor = sqrt(complex<float>(2.0f * M_PI, 0.0f) * z);
        
        for (int n = 0; n <= n_max; ++n) {
            result[n] = exp_z / sqrt_factor;
            // Apply correction factor for different orders (simplified)
            if (n > 0) {
                result[n] *= pow(complex<float>(1.0f, 0.0f) / z, complex<float>(n, 0.0f));
            }
        }
    }
    
    return result;
}

Tensor4D<complex<float>> SOAP0_local(
    const int npoints,
    const int lcut,
    const int mcut,
    const int natmax,
    const int nspecies,
    const vector<int>& nat,
    const Tensor3D<int>& nneigh,
    const Tensor4D<float>& efact,
    const Tensor6D<complex<float>>& sph_i6,
    const Tensor6D<complex<float>>& sph_j6,
    const vector<float>& divfac
) {
    // init memory
    auto skernel = Tensor4D<complex<float>>(
        npoints, Tensor3D<complex<float>>(
            npoints, vector<vector<complex<float>>>(
                natmax, vector<complex<float>>(
                    natmax, complex<float>(0.0f)
                )
            )
        )
    );

    // parallel computation
    auto parallel_ISOAP = [&](
        const int i,
        const int j, 
        const int ii,
        const int jj) -> Tensor4D<complex<float>> {

            // init memory for ISOAP
            Tensor4D<complex<float>> ISOAP(
                nspecies, 
                Tensor3D<complex<float>>(
                    lcut + 1, 
                    vector<vector<complex<float>>>(
                        mcut, 
                        vector<complex<float>>(mcut, complex<float>(0.0f))
                    )
                )
            );

            for (int ix = 0; ix < nspecies; ++ix) {

                // init memory for sph_in
                Tensor3D<complex<float>> sph_in(
                    nneigh[i][ii][ix], 
                    vector<vector<complex<float>>>(
                        nneigh[j][jj][ix], 
                        vector<complex<float>>(lcut + 1, complex<float>(0.0f))
                    )
                );

                for (int iii = 0; iii < nneigh[i][ii][ix]; ++iii) {
                    for (int jjj = 0; jjj < nneigh[j][jj][ix]; ++jjj) {
                        // Calculate spherical harmonics
                        sph_in[iii][jjj] = spherical_in(lcut, complex<float>(0.0f, 0.0f));
                    }
                }

                // Perform contraction over neighbour indexes (equivalent to np.einsum)
                // einsum: 'a,b,abl,alm,blk->lmk'
                // a: efact[i][ii][ix][0:nneigh[i][ii][ix]]
                // b: efact[j][jj][ix][0:nneigh[j][jj][ix]]  
                // abl: sph_in[:,:,:]
                // alm: sph_i6[i][ii][ix][0:nneigh[i][ii][ix]][:][:]
                // blk: sph_j6[j][jj][ix][0:nneigh[j][jj][ix]][:][:]
                // ->lmk: result ISOAP[ix][l][m][k]
                
                for (int l = 0; l <= lcut; ++l) {
                    for (int m = 0; m < mcut; ++m) {
                        for (int k = 0; k < mcut; ++k) {
                            complex<float> sum(0.0f, 0.0f);
                            
                            for (int a = 0; a < nneigh[i][ii][ix]; ++a) {
                                for (int b = 0; b < nneigh[j][jj][ix]; ++b) {
                                    // einsum contraction: efact[a] * efact[b] * sph_in[a][b][l] * sph_i6[a][l][m] * sph_j6[b][l][k]
                                    sum += efact[i][ii][ix][a] * efact[j][jj][ix][b] * 
                                           sph_in[a][b][l] * sph_i6[i][ii][ix][a][l][m] * sph_j6[j][jj][ix][b][l][k];
                                }
                            }
                            
                            ISOAP[ix][l][m][k] = sum;
                        }
                    }
                }

            }
            skernel[i][j][ii][jj] = combine_spectra(lcut, mcut, nspecies, ISOAP, divfac);
    };

    // get number of thread in this machine
    int nthreads = std::thread::hardware_concurrency();

    // calculate number of jobs
    unsigned long njobs = npoints * npoints * natmax * natmax;

    // create list of jobs
    vector<array<int, 4>> jobs_param;
    jobs_param.reserve(njobs);
    array<int, 4> job_temp({0,0,0,0});
    for (int i = 0; i < npoints; ++i) {
        for (int j = 0; j < npoints; ++j) {
            for (int ii = 0; ii < nat[i]; ++ii) {
                for (int jj = 0; jj < nat[j]; ++jj) {
                    job_temp[0] = i;
                    job_temp[1] = j;
                    job_temp[2] = ii;
                    job_temp[3] = jj;
                    jobs_param.push_back(job_temp);
                }
            }
        }
    }

    // update njobs
    njobs = jobs_param.size();

    // job allocator
    auto allocator = [&](int start, int end) {
        for (int i = start; i < end; ++i) {
            // allocate job to this thread
            parallel_ISOAP(jobs_param[i][0], jobs_param[i][1], jobs_param[i][2], jobs_param[i][3]);
        }
    };

    // calculate job size
    int job_size = njobs / nthreads;

    vector<std::thread> threads;

    for (int i = 0; i < nthreads; ++i) {
        int start = i * job_size;
        int end = (i == nthreads - 1) ? njobs : start + job_size;
        threads.emplace_back(allocator, start, end);
    }

    for (auto& thread : threads) {
        thread.join();
    }

}