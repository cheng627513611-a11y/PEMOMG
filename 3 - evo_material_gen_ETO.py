import os
import json
import time
import random
import pickle
import argparse
from collections import defaultdict
from copy import deepcopy
import numpy as np
import torch
import smact
from pymatgen.core import PeriodicSite
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from pymoo.core.sampling import Sampling
from simple_dist import sp_lookup
from pymatgen.core.structure import Structure

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice, Binary
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, \
    MixedVariableDuplicateElimination
from pymatgen.core.composition import ChemicalPotential, Composition
from pymoo.optimize import minimize

from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter
from pymatgen.analysis.phase_diagram import PDEntry
from pymatgen.core.periodic_table import Element

np.set_printoptions(precision=4, suppress=True)

# from scripts.compute_metrics import Crystal,GenEval
short_ele = ['Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
short_LaAc_id = {'Am': 66, 'Cm': 70, 'Bk': 67, 'Cf': 69, 'Es': 73, 'Fm': 75, 'Md': 81, 'No': 83, 'Lr': 79}
re_short_LaAc = {short_LaAc_id[k]: k for k in short_LaAc_id}
LaAc = ['Xe', 'Ac', 'Am', 'Bk', 'Ce', 'Cf', 'Cm', 'Dy', 'Er', 'Es', 'Eu', 'Fm', 'Gd', 'Ho', 'La', 'Lr', 'Lu', 'Md',
        'Nd', 'No', 'Np', 'Pa', 'Pm', 'Pr', 'Pu', 'Sm', 'Tb', 'Th', 'Tm', 'U', 'Yb']
LaAc_id = {'Xe': 64, 'Ac': 65, 'Am': 66, 'Bk': 67, 'Ce': 68, 'Cf': 69, 'Cm': 70, 'Dy': 71, 'Er': 72, 'Es': 73,
           'Eu': 74, 'Fm': 75, 'Gd': 76, 'Ho': 77, 'La': 78, 'Lr': 79, 'Lu': 80, 'Md': 81, 'Nd': 82, 'No': 83,
           'Np': 84, 'Pa': 85, 'Pm': 86, 'Pr': 87, 'Pu': 88, 'Sm': 89, 'Tb': 90, 'Th': 91, 'Tm': 92, 'U': 93,
           'Yb': 94}
re_LaAc = {LaAc_id[k]: k for k in LaAc_id}

# prepare elemente labels
with open('data/elements_id_NoPo.json', 'r') as f:
    e_d = json.load(f)
#    re = {e_d[k]: k for k in e_d}

#print(re)
print(re_LaAc)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

with open('data/paras.pickle', 'rb') as handle:
    AUX_DATA = pickle.load(handle)
n_spacegroup = len(AUX_DATA[-2])
box_abc_pat = torch.Tensor(AUX_DATA[0]).to(device)
box_angle_pat = torch.Tensor(AUX_DATA[1]).to(device)
cr = torch.Tensor(AUX_DATA[3]).to(device)
lr = torch.Tensor([AUX_DATA[2]]).to(device)
spinfo = sp_lookup(device, AUX_DATA[-2])
sp2id = AUX_DATA[-2]
id2sp = {sp2id[k]: k for k in sp2id}
print(id2sp)


def get_f_e_energy(comp_str, eles):
    with MPRester("your key") as mpr:
        es = [e.name for e in eles]
        # Obtain only corrected GGA and GGA+U ComputedStructureEntry objects
        entries = mpr.get_entries_in_chemsys(elements=es,
                                             additional_criteria={"thermo_types": ["GGA_GGA+U"]})
        # Construct phase diagram
        pd = PhaseDiagram(entries)

    energy_dict = {}
    for ele in eles:
        e = Element(ele.name)
        i = 0
        while i < len(entries):
            try:
                element_profile = pd.get_element_profile(element=e, comp=entries[i].composition)
                energy_dict[ele.name] = element_profile[0]['chempot']
                break
            except:
                i += 1
        if i == len(entries):
            return 100, 100

    cp = ChemicalPotential(energy_dict)
    comp = Composition(comp_str)
    energy = cp.get_energy(composition=comp)
    pdentry = PDEntry(composition=comp, energy=energy)
    fe = pd.get_form_energy(pdentry)
    eah = pd.get_e_above_hull(pdentry)
    return fe, eah

class StrucPost(Structure):
    def merge_same_element(self, prop=0.6):
        mapping = defaultdict(list)
        for site in self.sites:
            mapping[site.species.reduced_formula].append(site)

        new_sites = []
        elements = list(mapping.keys())
        for symbol in elements:
            holes = mapping[symbol]
            if len(holes) == 1:
                new_sites.append(holes[0])
                continue
            frac_coords = np.array([site.frac_coords for site in holes])
            d = self.lattice.get_all_distances(frac_coords, frac_coords)
            np.fill_diagonal(d, 0)
            clusters = fcluster(linkage(squareform((d + d.T) / 2)), float(holes[0].specie.atomic_radius) * 2 * prop,
                                "distance")

            for c in np.unique(clusters):
                inds = np.where(clusters == c)[0]
                species = holes[inds[0]].species
                coords = holes[inds[0]].frac_coords
                for n, i in enumerate(inds[1:]):
                    offset = holes[i].frac_coords - coords
                    coords = coords + ((offset - np.round(offset)) / (n + 2)).astype(coords.dtype)
                new_sites.append(PeriodicSite(species, coords, self.lattice))
        self._sites = new_sites

    def check_dist(self, prop=0.75):
        d = self.distance_matrix
        iu = np.triu_indices(len(d), 1)
        d = d[iu]

        atom_radius = []
        for i in range(len(self)):
            for j in range(i + 1, len(self)):
                atom_radius.append(self[i].specie.atomic_radius + self[j].specie.atomic_radius)
        atom_radius = np.array(atom_radius) * prop

        # return np.all(d > atom_radius)
        return np.all(d > atom_radius), min(d), max(d)

    def get_overall_charge(self):
        composition = self.composition
        all_sites_elements = [site.specie for site in self.sites]
        space = smact.element_dictionary(all_sites_elements)
        element_charges = {}
        for element in composition:
            element_charges[element.symbol] = element.oxi_state

        # Calculate the overall charge of the structure
        overall_charge = sum(element_charges[element.symbol] * composition[element] for element in composition)
        return overall_charge

def from_solution(spid, coords, lengths, angles, elements):
    f = open('data/cif-template.txt', 'r')
    template = f.read()
    f.close()

    template = template.replace('SYMMETRY-SG', id2sp[spid])
    template = template.replace('LAL', str(lengths[0]))
    template = template.replace('LBL', str(lengths[1]))
    template = template.replace('LCL', str(lengths[2]))
    template = template.replace('DEGREE1', str(angles[0]))
    template = template.replace('DEGREE2', str(angles[1]))
    template = template.replace('DEGREE3', str(angles[2]))
    f = open('data/symmetry-equiv/%s.txt' % id2sp[spid].replace('/', '#'), 'r')
    sym_ops = f.read()
    f.close()

    template = template.replace('TRANSFORMATION\n', sym_ops)

    for m in range(3):
        row = ['', re_LaAc[elements[m]], re_LaAc[elements[m]] + str(m), \
               str(coords[m][0]), str(coords[m][1]), str(coords[m][2]), '1']
        row = '  '.join(row) + '\n'
        template += row

    template += '\n'
    try:
        crystal = StrucPost.from_str(template, fmt="cif")
        formula = crystal.composition.reduced_formula
        # print(formula)
        sg_info = crystal.get_space_group_info(symprec=0.1)
        # print(sg_info)
        return crystal
    except Exception as e:
        # print(e)
        raise e


class MaterialGeneration(ElementwiseProblem):
    def __init__(self, **kwargs):
        variables = dict()

        variables[f"space_group_id"] = Integer(bounds=(0, kwargs['n_spacegroup'] - 1))

        for k in range(3):
            variables[f"ele_id{k:01}"] = Choice(options=kwargs['ele_list'])

        for k in range(9):
            variables[f"coordinate{k:01}"] = Real(bounds=(-1, 1))

        for k in range(3):
            variables[f"angle{k:01}"] = Real(bounds=(-1, 1))
        self.box_abc_pat = kwargs['box_abc_pat']
        self.box_angle_pat = kwargs['box_angle_pat']
        self.cr = kwargs['cr']
        self.lr = kwargs['lr']
        self.device = kwargs['device']
        super().__init__(vars=variables, n_obj=4, n_ieq_constr=3, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        space_group_id = X["space_group_id"]
        ele_id0 = X["ele_id0"]
        ele_id1 = X["ele_id1"]
        ele_id2 = X["ele_id2"]
        coordinate0 = X["coordinate0"]
        coordinate1 = X["coordinate1"]
        coordinate2 = X["coordinate2"]
        coordinate3 = X["coordinate3"]
        coordinate4 = X["coordinate4"]
        coordinate5 = X["coordinate5"]
        coordinate6 = X["coordinate6"]
        coordinate7 = X["coordinate7"]
        coordinate8 = X["coordinate8"]
        angle0 = X["angle0"]
        angle1 = X["angle1"]
        angle2 = X["angle2"]

        c012 = [coordinate0, coordinate1, coordinate2]
        c345 = [coordinate3, coordinate4, coordinate5]
        c678 = [coordinate6, coordinate7, coordinate8]
        coords = torch.Tensor([c012, c345, c678]).to(device)
        arr_coords = coords * self.cr + self.cr
        arr_coords = arr_coords.cpu().detach().numpy()
        arr_coords = np.round(arr_coords, 2)

        box_abc = torch.Tensor([angle0, angle1, angle2]).to(device)
        box_abc = torch.exp(box_abc * self.lr + self.lr)
        # box_abc = torch.einsum('bl,bls->bs', box_abc, self.box_abc_pat[space_group_id])
        pat = self.box_abc_pat[space_group_id].unsqueeze(0)
        box_abc = torch.einsum('bl,bls->bs', box_abc, pat)
        box_angles = self.box_angle_pat[space_group_id]

        arr_lengths = box_abc[0].cpu().detach().numpy()
        arr_angles = box_angles.cpu().detach().numpy()
        arr_ele = np.array([ele_id0, ele_id1, ele_id2])

        # Objectives:
        # formation energy
        # energy above hull
        # minimize min atoms length
        # maximize max atoms length
        # 
        # Constraints:
        # 0 different atoms
        # 1 cifs
        # 2 distance greater than 0.75
        G = [0, 0, 0]
        element_list = [ele_id0, ele_id1, ele_id2]
        # G[0] = (sum(arr_lengths) - 15) > 0
        G[0] = not(len(element_list) == len(set(element_list)))
        try:
            crystal = from_solution(space_group_id, arr_coords, arr_lengths, arr_angles, arr_ele)
        except:
            G[1] = 1
        try:
            G[2], min_atom, max_atom = crystal.check_dist()
        except:
            min_atom = 0
            max_atom = 100
        try:
            form_e, e_above = get_f_e_energy(crystal.composition, crystal.elements)
        except:
            form_e = 100
            e_above = 100

        out["F"] = [form_e, e_above, -min_atom, max_atom]
        out["G"] = G

        return form_e, e_above


def save_solutions(gen, Xs, lr, cr, box_abc_pat, box_angle_pat):
    available = 0
    for count_sample, X in enumerate(Xs):
        if save_x(gen, X, count_sample, lr, cr, box_abc_pat, box_angle_pat) is not None:
            available += 1
    return available

def save_x(gen, X, count_sample, lr, cr, box_abc_pat, box_angle_pat):
    space_group_id = X["space_group_id"]
    ele_id0 = X["ele_id0"]
    ele_id1 = X["ele_id1"]
    ele_id2 = X["ele_id2"]
    coordinate0 = X["coordinate0"]
    coordinate1 = X["coordinate1"]
    coordinate2 = X["coordinate2"]
    coordinate3 = X["coordinate3"]
    coordinate4 = X["coordinate4"]
    coordinate5 = X["coordinate5"]
    coordinate6 = X["coordinate6"]
    coordinate7 = X["coordinate7"]
    coordinate8 = X["coordinate8"]
    angle0 = X["angle0"]
    angle1 = X["angle1"]
    angle2 = X["angle2"]

    c012 = [coordinate0, coordinate1, coordinate2]
    c345 = [coordinate3, coordinate4, coordinate5]
    c678 = [coordinate6, coordinate7, coordinate8]
    coords = torch.Tensor([c012, c345, c678]).to(device)
    arr_coords = coords * cr + cr
    arr_coords = arr_coords.cpu().detach().numpy()
    arr_coords = np.round(arr_coords, 2)

    box_abc = torch.Tensor([angle0, angle1, angle2]).to(device)
    box_abc = torch.exp(box_abc * lr + lr)
    pat = box_abc_pat[space_group_id].unsqueeze(0)
    box_abc = torch.einsum('bl,bls->bs', box_abc, pat)
    box_angles = box_angle_pat[space_group_id]

    arr_lengths = box_abc[0].cpu().detach().numpy()
    arr_angles = box_angles.cpu().detach().numpy()
    arr_ele = np.array([ele_id0, ele_id1, ele_id2])

    crystal = from_solution(space_group_id, arr_coords, arr_lengths, arr_angles, arr_ele)
    try:
        form_e, e_above = get_f_e_energy(crystal.composition, crystal.elements)
    except:
        form_e = 100
        e_above = 100

    cry = save_cif(gen, count_sample, space_group_id, arr_coords, arr_lengths, arr_angles, arr_ele, form_e, e_above)
    return cry


def save_cif(gen, count_sample, spid, coords, lengths, angles, elements, form_e, e_above):
    f = open('data/cif-template.txt', 'r')
    template = f.read()
    f.close()

    template = template.replace('SYMMETRY-SG', id2sp[spid])
    template = template.replace('LAL', str(lengths[0]))
    template = template.replace('LBL', str(lengths[1]))
    template = template.replace('LCL', str(lengths[2]))
    template = template.replace('DEGREE1', str(angles[0]))
    template = template.replace('DEGREE2', str(angles[1]))
    template = template.replace('DEGREE3', str(angles[2]))
    f = open('data/symmetry-equiv/%s.txt' % id2sp[spid].replace('/', '#'), 'r')
    sym_ops = f.read()
    f.close()

    template = template.replace('TRANSFORMATION\n', sym_ops)

    for m in range(3):
        row = ['', re_LaAc[elements[m]], re_LaAc[elements[m]] + str(m), \
               str(coords[m][0]), str(coords[m][1]), str(coords[m][2]), '1']
        row = '  '.join(row) + '\n'
        template += row

    template += f'\n# Formation Energy: {form_e}\n'
    template += f'# Energy Above Hull: {e_above}\n'
    template += '\n'

    dir = 'evo_gen/gen-%d/' % (gen)
    if not os.path.exists(dir):
        os.makedirs(dir)
    f = open('%s%s---%d.cif' % \
             (dir, id2sp[spid].replace('/', '#'), count_sample), 'w')
    f.write(template)
    f.close()
    try:
        crystal = Structure.from_str(template, fmt="cif")
        # formula = crystal.composition.reduced_formula
        # print(formula)
        # sg_info = crystal.get_space_group_info(symprec=0.1)
        # print(sg_info)
        return crystal
    except Exception as e:
        # print(e)
        return None


def replace_elements_with_new(population, re_LaAc):
    new_population = []
    for X in population:
        new_X = deepcopy(X)
        new_ele_ids = random.sample(list(re_LaAc.keys()), 3)
        new_X["ele_id0"] = new_ele_ids[0]
        new_X["ele_id1"] = new_ele_ids[1]
        new_X["ele_id2"] = new_ele_ids[2]
        new_population.append(new_X)
    new_population = np.array(new_population)
    return new_population


class CustomSampling(Sampling):
    def __init__(self, initial_population):
        super().__init__()
        self.initial_population = initial_population

    def _do(self, problem, n_samples, **kwargs):
        if n_samples > len(self.initial_population):
            raise ValueError("Number of samples requested exceeds the size of the initial population.")
        return self.initial_population[:n_samples]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crystal Structure Generative Adversial Network')
    parser.add_argument("--element_crystal", type=int, default=3, help="number of elements in the crystal")
    parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
    parser.add_argument("--n_samples", type=int, default=10000, help="number of crystal to generate")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    parser.add_argument('--gpu_id', type=int, default=8)
    parser.add_argument('--savedir', type=str, default=None)
    parser.add_argument('--matdir', type=str, default='virtual_mat')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false')
    parser.set_defaults(verbose=True)
    args = parser.parse_args()
    
    argdict = vars(args)
    print(argdict)

    with open('short_problem_results.pickle', 'rb') as f:
        initial_population = pickle.load(f)
    new_initial_population = replace_elements_with_new(initial_population, re_LaAc)
    with open('long_cycle_initial_population.pickle', 'wb') as file:
        pickle.dump(new_initial_population, file)


    total_gen = 100
    pop_size = 100
    ele_list = [i for i in re_LaAc.keys()]
    problem = MaterialGeneration(n_spacegroup=n_spacegroup,
                                 box_abc_pat=box_abc_pat,
                                 box_angle_pat=box_angle_pat,
                                 cr=cr,
                                 lr=lr,
                                 ele_list=ele_list,
                                 device=device)

    # for name, var in problem.vars.items():
    #     print(name, var)

    algorithm = NSGA2(pop_size=pop_size,
                      sampling=CustomSampling(new_initial_population),
                      mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                      eliminate_duplicates=MixedVariableDuplicateElimination(),
                      )

    algorithm.setup(problem, termination=('n_gen', total_gen), seed=40, verbose=False)
    log = open(f'evo_gen2.txt', 'a+')
    print('Generation\tFeasible Rate\tCIF', file=log)
    # until the algorithm has no terminated
    start = time.time()
    while algorithm.has_next():
        # do the next iteration
        algorithm.next()
        # do same more things, printing, logging, storing or even modifying the algorithm object
        print(algorithm.n_gen, algorithm.evaluator.n_eval)

        if algorithm.n_gen % 1 == 0:
            res = algorithm.result()
            if res is not None and res.X is not None:
                available = save_solutions(algorithm.n_gen, res.X, lr, cr, box_abc_pat, box_angle_pat)
                print('%d\t%2f\t%2f' % (algorithm.n_gen, len(res.X) / pop_size, available / len(res.X)), file=log)
                with open(f'./evo_gen/res_x_{algorithm.n_gen}.pickle', "wb") as file:
                    # Dump the data into the file
                    pickle.dump(res.X, file)
                with open(f'./evo_gen/res_f_{algorithm.n_gen}.pickle', "wb") as file:
                    # Dump the data into the file
                    pickle.dump(res.F, file)
            else:
                print('Generation: %d, Feasible Rate: 0, CIF: 0' % (algorithm.n_gen), file=log)
    print(f'Evolve took {time.time() - start:.2f} seconds', file=log)

