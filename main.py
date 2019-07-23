import numpy as np
from skimage.io import imread, imsave


def quilting(input_img, patch_size):
    input_h, input_w, input_c = input_img.shape
    valid_input_h, valid_input_w = input_h - patch_size, input_w - patch_size
    target_h, target_w = input_h * 2, input_w * 2
    overlap = np.int(patch_size / 6)
    tolerance = 0.1
    patch_no_overlap = patch_size - overlap
    quilting_h = np.int(np.ceil((target_h - patch_size) / patch_no_overlap) * patch_no_overlap) + patch_size
    quilting_w = np.int(np.ceil((target_w - patch_size) / patch_no_overlap) * patch_no_overlap) + patch_size
    output = np.zeros((quilting_h, quilting_w, input_c), dtype='float32')

    for i in range(0, quilting_h - patch_size + 1, patch_no_overlap):
        for j in range(0, quilting_w - patch_size + 1, patch_no_overlap):
            # print(i, j)
            if i == 0 and j == 0:
                rand_h = np.random.randint(0, input_h - patch_size, 1)[0]
                rand_w = np.random.randint(0, input_w - patch_size, 1)[0]
                firstPatch = input_img[rand_h:rand_h + patch_size, rand_w:rand_w + patch_size]
                output[:patch_size, :patch_size] = firstPatch
            elif i == 0:
                best_patch_h, best_patch_w, _ = findBestHorizontalPatch(valid_input_h, valid_input_w, input_c, i, j, patch_size, overlap, input_img, output, tolerance)
                cutOverlapHorizontal(input_img, best_patch_h, best_patch_w, patch_size, overlap, i, j, output)
                
            elif j == 0:
                best_patch_h, best_patch_w, _ = findBestVerticalPatch(valid_input_h, valid_input_w, input_c, i, j, patch_size, overlap, input_img, output, tolerance)
                cutOverlapVertical(input_img, best_patch_h, best_patch_w, patch_size, overlap, i, j, output)
            else:
                best_patch_h, best_patch_w, _ = findBestBothSidePatch(valid_input_h, valid_input_w, input_c, i, j, patch_size, overlap, input_img, output, tolerance)
                cutBoth(input_img, best_patch_h, best_patch_w, patch_size, overlap, i, j, output)
                
    return output

def transfer(source_img, target_img, iter, patch_size, alpha, tolerance):
    input_h, input_w, input_c = source_img.shape
    valid_input_h, valid_input_w = input_h - patch_size, input_w - patch_size
    target_h, target_w = target_img.shape[0], target_img.shape[1]

    overlap = np.int(patch_size / 6)
    patch_no_overlap = patch_size - overlap
    quilting_h = np.int(np.ceil((target_h - patch_size) / patch_no_overlap) * patch_no_overlap) + patch_size
    quilting_w = np.int(np.ceil((target_w - patch_size) / patch_no_overlap) * patch_no_overlap) + patch_size
    output = np.zeros((quilting_h, quilting_w, input_c), dtype='float32')

    for i in range(0, quilting_h - patch_size + 1, patch_no_overlap):
        for j in range(0, quilting_w - patch_size + 1, patch_no_overlap):
            # print(i, j)
            cost_matrix_target = get_cost_matrix_target(valid_input_h, valid_input_w, input_c, patch_size, target_img, source_img, i, j)
            if i == 0 and j == 0:
                best_patch_h, best_patch_w = random_pick(cost_matrix_target, tolerance)
                output[:patch_size, :patch_size] = source_img[best_patch_h: best_patch_h + patch_size, best_patch_w: best_patch_w + patch_size]
            elif i == 0:
                _, _, cost_matrix_hor = findBestHorizontalPatch(valid_input_h, valid_input_w, input_c, i, j, patch_size, overlap, source_img, output, tolerance)
                sum_cost_matrix = alpha * cost_matrix_hor + (1 - alpha) * cost_matrix_target
                best_patch_h, best_patch_w = random_pick(sum_cost_matrix, tolerance)
                cutOverlapHorizontal(source_img, best_patch_h, best_patch_w, patch_size, overlap, i, j, output)
            elif j == 0:
                _, _, cost_matrix_ver = findBestHorizontalPatch(valid_input_h, valid_input_w, input_c, i, j, patch_size, overlap, source_img, output, tolerance)
                sum_cost_matrix = alpha * cost_matrix_ver + (1 - alpha) * cost_matrix_target
                best_patch_h, best_patch_w = random_pick(sum_cost_matrix, tolerance)
                cutOverlapVertical(source_img, best_patch_h, best_patch_w, patch_size, overlap, i, j, output)
            else:
                _, _, total_cost_matrix = findBestBothSidePatch(valid_input_h, valid_input_w, input_c, i, j, patch_size, overlap, source_img, output, tolerance)
                sum_cost_matrix = alpha * total_cost_matrix + (1 - alpha) * cost_matrix_target
                best_patch_h, best_patch_w = random_pick(sum_cost_matrix, tolerance)
                cutBoth(source_img, best_patch_h, best_patch_w, patch_size, overlap, i, j, output)
    return output

#the defualt return is the best patch index
def findBestHorizontalPatch(valid_input_h, valid_input_w, input_c, i, j, patch_size, overlap, input_img, output, tolerance):
    #find the best match patch
    cost_matrix = np.zeros((valid_input_h, valid_input_w, input_c))
    for y_overlap in range(patch_size):
        for x_overlap in range(overlap):
            cost = input_img[y_overlap:y_overlap + valid_input_h, x_overlap:x_overlap + valid_input_w] - output[i + y_overlap, j + x_overlap]
            cost_matrix += np.square(cost)
    cost_matrix = np.sum(cost_matrix, axis=2)
    best_patch_h, best_patch_w = random_pick(cost_matrix, tolerance)

    return best_patch_h, best_patch_w, cost_matrix

#the defualt return is the best patch index
def findBestVerticalPatch(valid_input_h, valid_input_w, input_c, i, j, patch_size, overlap, input_img, output, tolerance):
    cost_matrix = np.zeros((valid_input_h, valid_input_w, input_c))
    for y_overlap in range(overlap):
        for x_overlap in range(patch_size):
            y_overlap_on_output = i + y_overlap
            x_overlap_on_output = j + x_overlap
            cost = input_img[y_overlap:y_overlap + valid_input_h, x_overlap:x_overlap +valid_input_w] - output[y_overlap_on_output, x_overlap_on_output]
            cost_matrix += np.square(cost)
    cost_matrix = np.sum(cost_matrix, axis=2)
    perfectIndex = np.where(cost_matrix == 0)
    cost_matrix[perfectIndex] = np.min(cost_matrix[np.where(cost_matrix > 0)])
    best_patch_h, best_patch_w = random_pick(cost_matrix, tolerance)
   
    return best_patch_h, best_patch_w, cost_matrix


def findBestBothSidePatch(valid_input_h, valid_input_w, input_c, i, j, patch_size, overlap, input_img, output, tolerance):
    _, _, vertical_cost_matrix = findBestVerticalPatch(valid_input_h, valid_input_w, input_c, i, j, patch_size, overlap, input_img, output, tolerance)
    _, _, horizontal_cost_matrix = findBestHorizontalPatch(valid_input_h, valid_input_w, input_c, i, j, patch_size, overlap, input_img, output, tolerance)
    perfectIndexh = np.where(horizontal_cost_matrix == 0)
    horizontal_cost_matrix[perfectIndexh] = np.min(horizontal_cost_matrix[np.where(horizontal_cost_matrix > 0)])
    perfectIndexv = np.where(vertical_cost_matrix == 0)
    vertical_cost_matrix[perfectIndexv] = np.min(vertical_cost_matrix[np.where(vertical_cost_matrix > 0)])
    total_cost_matrix = horizontal_cost_matrix + vertical_cost_matrix
    best_patch_h, best_patch_w = random_pick(total_cost_matrix, tolerance)
    
    return best_patch_h, best_patch_w, total_cost_matrix


def cutOverlapHorizontal(input_img, best_patch_h, best_patch_w, patch_size, overlap, i, j, output):
    #find min cost path
    overlap_input = input_img[best_patch_h: best_patch_h + patch_size, best_patch_w: best_patch_w + overlap]
    overlap_output = output[i: i + patch_size, j: j + overlap]
    overlap_errors = np.square(overlap_input - overlap_output)
    E = overlap_errors.copy()
    for ei in range(1, patch_size):
        for ej in range(1, overlap - 1):
            E[ei, ej] = overlap_errors[ei, ej] + np.min(E[ei - 1, ej - 1: ej + 1])
    E = np.mean(E, axis=2)
    minCostPath = np.argmin(E, axis=1)
    for edgei in range(len(minCostPath)):
        minIndex = minCostPath[edgei]
        output[i + edgei, j + minIndex: j + patch_size] = input_img[best_patch_h + edgei, best_patch_w + minIndex: best_patch_w + patch_size]


def cutOverlapVertical(input_img, best_patch_h, best_patch_w, patch_size, overlap, i, j, output):
     #find min cost path
    overlap_input = input_img[best_patch_h: best_patch_h + overlap, best_patch_w: best_patch_w + patch_size]
    overlap_output = output[i: i + overlap, j: j + patch_size]
    overlap_errors = np.square(overlap_input - overlap_output)
    E = overlap_errors.copy()
    for ei in range(1, overlap - 1):
        for ej in range(1, patch_size):
            E[ei, ej] = overlap_errors[ei, ej] + np.min(E[ei - 1: ei + 1, ej - 1])
    E = np.mean(E, axis=2)
    minCostPath = np.argmin(E, axis=0)
    for edgej in range(len(minCostPath)):
        minIndex = minCostPath[edgej]
        output[i + minIndex: i + patch_size, j + edgej] = input_img[best_patch_h + minIndex: best_patch_h + patch_size, best_patch_w + edgej]

def random_pick(cost_matrix, tolerance):
    perfectIndex = np.where(cost_matrix == 0)
    cost_matrix[perfectIndex] = np.min(cost_matrix[np.where(cost_matrix > 0)])
    tolerant_index = np.where(cost_matrix <= np.min(cost_matrix) * (1 + tolerance))
    random_pick = np.random.choice(range(len(tolerant_index[0])), 1)
    best_patch_h, best_patch_w = tolerant_index[0][random_pick], tolerant_index[1][random_pick]
    best_patch_h, best_patch_w = best_patch_h[0], best_patch_w[0]
    return best_patch_h, best_patch_w


def get_cost_matrix_target(valid_input_h, valid_input_w, input_c, patch_size, target_img, source_img, i, j):
    target_h, target_w = target_img.shape[0], target_img.shape[1]
    cost_matrix = np.zeros((valid_input_h, valid_input_w, input_c))
    for pi in range(patch_size):
        for pj in range(patch_size):
            th = np.min((i + pi, target_h - 1))
            tw = np.min((j + pj, target_w - 1))
            target_pixel = target_img[th, tw]
            source_img_valid = source_img[pi: pi + valid_input_h, pj: pj + valid_input_w]
            cost_matrix += np.square(source_img_valid - target_pixel)
    cost_matrix = np.sum(cost_matrix, axis=2)
    return cost_matrix


def cutBoth(input_img, best_patch_h, best_patch_w, patch_size, overlap, i, j, output):
    #h
    overlap_input = input_img[best_patch_h: best_patch_h + patch_size, best_patch_w: best_patch_w + overlap]
    overlap_output = output[i: i + patch_size, j: j + overlap]
    overlap_errors = np.square(overlap_input - overlap_output)
    E = overlap_errors.copy()
    for ei in range(1, patch_size):
        for ej in range(1, overlap - 1):
            E[ei, ej] = overlap_errors[ei, ej] + np.min(E[ei - 1, ej - 1: ej + 1])
    E = np.mean(E, axis=2)
    minCostPath = np.argmin(E, axis=1)
    overlap_input_v = input_img[best_patch_h: best_patch_h + overlap, best_patch_w + overlap: best_patch_w + patch_size]
    overlap_output_v = output[i: i + overlap, j + overlap: j + patch_size]
    overlap_errors_v = np.square(overlap_input_v - overlap_output_v)
    Ev = overlap_errors_v.copy()
    for evi in range(1, overlap - 1):
        for evj in range(1, patch_size - overlap):
            Ev[evi, evj] = overlap_errors_v[evi, evj] + np.min(Ev[evi - 1: evi + 1, evj - 1])
    Ev = np.mean(Ev, axis=2)
    minCostPathVertical = np.argmin(Ev, axis=0)
    for edgei in range(overlap, len(minCostPath)):
        minIndex = minCostPath[edgei]
        output[i + edgei, j + minIndex: j + patch_size] = input_img[best_patch_h + edgei, best_patch_w + minIndex: best_patch_w + patch_size]
    for edgej in range(overlap, len(minCostPathVertical)):
        minIndex = minCostPathVertical[edgej]
        output[i + minIndex: i + patch_size, j + edgej] = input_img[best_patch_h + minIndex: best_patch_h + patch_size, best_patch_w + edgej]

from os.path import normpath as fn  # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

# img = np.float32(imread(fn('inputs/cotton.png')))

# patch_size = 16
# output = quilting(img, patch_size)

# imsave(fn('outputs/output_cotton.png'), output / 255)

source_img = np.float32(imread(fn('inputs/rice.png')))
target_img = np.float32(imread(fn('inputs/man_face.png')))
patch_size = 48
for i in range(0, 5):
    reduce_ratio = float(2) / float(3)
    reduced_patch_size = int(patch_size * (reduce_ratio ** i))
    print(reduced_patch_size)
    output = transfer(source_img, target_img, False, reduced_patch_size, 0.2, 0.1)
    imsave(fn('outputs/output_manface_itr%i_alpha0.2.png'%(i)), output / 255)
