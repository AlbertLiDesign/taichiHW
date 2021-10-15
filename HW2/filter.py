#
# def preFlt():
#     centers = np.zeros((nelx * sub_res * nely * sub_res, 2))
#     for i in range(nelx):
#         for j in range(nely):
#             for subx in range(sub_res):
#                 for suby in range(sub_res):
#                     centers[(j * sub_res + suby) * nelx * sub_res +  i * sub_res + subx, 0] = j * sub_res + suby
#                     centers[(j * sub_res + suby) * nelx * sub_res +  i * sub_res + subx, 1] = i * sub_res + subx
#
#     # Weighting factors
#     kdtree = spatial.KDTree(centers)
#
#     Fm={}
#     for i in range(nelx):
#         for j in range(nely):
#             for subx in range(sub_res):
#                 for suby in range(sub_res):
#                     index= (j * sub_res + suby) * nelx * sub_res + i * sub_res + subx
#                     Fm[index] = [[],[]]
#                     dist, id = kdtree.query(centers, distance_upper_bound=rmin, workers=-1)
#                     for a in range(len(id)):
#                         if (dist[a]  > 0) & (dist[a] < rmin):
#                             Fm[index][0].append(id[a])
#                             Fm[index][1].append(rmin - dist[a])
#                     Fm[index][1] = np.divide(Fm[index][1], np.sum(Fm[index][1]))
#     return Fm
#     # for i in range(len(elm)):
#     #     Fm[elm[i]] = [[], []]
#     #     for j in range(len(id[i])):
#     #         if(dist[i][j] < Rmin):
#     #             # Save indices
#     #             Fm[elm[i]][0].append(id[i][j])
#     #             # Save distance
#     #             Fm[elm[i]][1].append(Rmin - dist[i][j])
#     #     # Calculate and save weights
#     #     Fm[elm[i]][1] = np.divide(Fm[elm[i]][1], np.sum(Fm[elm[i]][1]))
#     # end1 = time.time()
#     # print("Cal Weights: " + str(end1-start1))