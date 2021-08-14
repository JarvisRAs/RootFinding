from yroots.polynomial import MultiPower
import pickle
import yroots as yr
import numpy as np
import time

a = -np.ones(3)
b = np.ones(3)
num_loops = 1
num_tests = 100
start_deg = 2
larger_deg = 20
deg_skip = 1

# timing_dict = np.load('YRoots_2D_poly_timings_no_sign_change.pkl', allow_pickle=True)
timing_dict = {i:float() for i in range(start_deg, larger_deg + 1)}
all_num_roots = []
all_avg_resids = []
all_max_resids = []
has_roots = False

for deg in range(start_deg, larger_deg + 1, deg_skip):
    coeffs = np.load("tests/random_tests/coeffs/dim3_deg{}_randn.npy".format(deg))
    deg_time = 0
    num_roots_deg = []
    avg_resids_per_deg = []
    max_resids_per_deg = []

    for test in range(num_tests):

        c1 = coeffs[test, 0, :, :, :]
        c2 = coeffs[test, 1, :, :, :]
        c3 = coeffs[test, 2, :, :, :]
        c1[0,0,0] = 0
        c2[0,0,0] = 0
        c3[0,0,0] = 0

        f = MultiPower(c1)
        g = MultiPower(c2)
        h = MultiPower(c3)

        test_time = 0
        for _ in range(num_loops):
            has_roots = False
            print("Degree {}, Test {}/{}".format(deg, test + 1, num_tests))
            start = time.time()
            roots = (yr.solve([f,g,h], a, b, target_deg = 1))
            end = time.time()
            num_roots = len(roots)
            num_roots_deg.append(num_roots)
            if num_roots > 0:
                has_roots = True
#            if len(roots) == 0:
#                print("!!! DIDNT FIND ANY ROOTS!!")

            if has_roots:
                norm_f =sum(sum(sum(abs(c1))))
                norm_g = sum(sum(sum(abs(c2))))
                norm_h = sum(sum(sum(abs(c3))))
                f_rel_resids = []
                g_rel_resids = []
                h_rel_resids = []
                for i in range(num_roots):
                    f_rel_resids = (abs(f.__call__(roots[i]))/norm_f)
                    g_rel_resids = (abs(g.__call__(roots[i]))/norm_g)
                    h_rel_resids = (abs(h.__call__(roots[i]))/norm_h)
                avg_resid_system = (sum(f_rel_resids) + sum(g_rel_resids) + sum(h_rel_resids))/(3*num_roots)
                max_resid_system = max(max(f_rel_resids), max(g_rel_resids), max(h_rel_resids))

            if has_roots:
                avg_resids_per_deg.append(avg_resid_system)
                max_resids_per_deg.append(max_resid_system)

            test_time += end - start

        if has_roots:
            avg_resid_per_deg = np.mean(avg_resids_per_deg)
            max_resid_per_deg = max(max_resids_per_deg)

        del c1, c2, c3, f, g, h

        deg_time += test_time/num_loops

    if has_roots:
        all_avg_resids.append(avg_resid_per_deg)
        all_max_resids.append(max_resid_per_deg)

    all_num_roots.append(num_roots_deg)

    timing_dict[deg] = deg_time/num_tests


    with open('YRoots_Old_no_macaulay_rand_poly_timings_dim_3.pkl', 'wb') as ofile:
        pickle.dump(timing_dict, ofile)

    np.save("YRoots_Old_checks_no_macaulay_avg_resids_rand_poly_dim_3", all_avg_resids, allow_pickle=True, fix_imports=True)
    np.save("YRoots_Old_checks_no_macaulay_max_resids_rand_poly_dim_3", all_max_resids, allow_pickle=True, fix_imports=True)
    np.save("YRoots_Old_checks_no_macaulay_num_roots_rand_poly_dim_3", all_num_roots, allow_pickle=True, fix_imports=True)

    del coeffs
    print("Degree {} takes on average {}s".format(deg, timing_dict[deg]))

