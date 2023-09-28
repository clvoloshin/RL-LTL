from asyncio.proactor_events import _ProactorDuplexPipeTransport
from os import terminal_size
from numpy.lib.arraysetops import isin
from flloat.parser.ltlf import LTLfParser
import flloat
import spot
import numpy as np
from typing import Dict, Set, Union
import buddy
from utls.utls import remove_from_str, timeit
from functools import lru_cache
from subprocess import check_output
from sys import platform
import os
import random
import re
from itertools import chain, combinations
from utls.utls import parse_stl_into_tree
import networkx as nx

# from helpers.str_utils import remove_fr
# ls import parse_stl_into_treeom_str
# from interfaces.automata import Label, LABEL_TRUE
# from interfaces.expr import Signal

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    pset = []
    s = list(iterable)
    pset.extend(list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1))))
    return [set(x) for x in pset]

class AdditionalConstraint(object):
    pass

class UpToN(AdditionalConstraint):
    def __init__(self, objs, num):
        self.num = num
        self.objs = objs
    
    def satisfied(self, constraint):
        if isinstance(constraint.condition_dict, dict):
            total = sum([constraint.condition_dict.get(obj, 0) for obj in self.objs])
            if total > self.num:
                return False
        return True

class AtLeastN(AdditionalConstraint):
    def __init__(self, objs, num):
        self.num = num
        self.objs = objs
    
    def satisfied(self, constraint):
        if isinstance(constraint.condition_dict, dict):
            total = sum([constraint.condition_dict.get(obj, 0) for obj in self.objs])
            if total <= self.num:
                return False
        return True

class Constants(AdditionalConstraint):
    def __init__(self, constants):
        self.constants = constants
    
    def satisfied(self, constraint):
        if isinstance(constraint.condition_dict, dict):
            for constant, value in self.constants.items():
                if constant not in constraint.condition_dict: continue
                if constraint.condition_dict[constant] != value: return False
        return True

def parse_bdd(bdd, d): #(bdd:buddy.bdd, d:spot.bdd_dict):
    """ Special cases: empty set for false, {LABEL_TRUE} for true """
    s = spot.bdd_format_set(d, bdd)
    # s is like: <cancel:0, grant:0, req:0, go:0><cancel:1, grant:0, req:1, go:0>

    if s == 'F':
        return {'False': 0} #set()
    if s == 'T':
        return {'True': 1} #{Label(dict())}

    cube_tokens = s.split('><')
    cube_tokens = map(lambda ct: remove_from_str(ct, '>< '), cube_tokens)
    cube_labels = {}
    for ct in cube_tokens:
        # cancel:0, grant:0, req:0, go:0
        lit_tokens = ct.split(',')
        lit_tokens = map(lambda lt: remove_from_str(lt, ', '), lit_tokens)
        
        clause = []
        clause_as_dict = {}
        for sig_name, sig_val in map(lambda tok: tok.split(':'), lit_tokens):
            if bool(int(sig_val)):
                clause.append(sig_name)
                clause_as_dict[sig_name] = 1
            else:
                clause.append('!' + sig_name)
                clause_as_dict[sig_name] = 0
        
        cube_labels[' && '.join(clause)] = clause_as_dict
    # return '|| '.join(cube_labels)
    return cube_labels


class AutomatonEdge(object):
    def __init__(self, parent, condition, condition_dict, child, acc_frontier=None):
        self.parent = parent
        self.condition = condition
        self.condition_dict = condition_dict
        self.child = child
        self.acc_frontier = acc_frontier

        self.finite = set()
        self.infinite = set()
        try:
            self.stl = parse_stl_into_tree(str(self.condition))
        except:
            import pdb; pdb.set_trace()
            parse_stl_into_tree(str(self.condition))
    
    def set_accepting(self, i, fin_or_inf):
        if fin_or_inf:
            self.infinite.add(i)
        else:
            self.finite.add(i)
        
    def __repr__(self) -> str:
        return "Edge(%s, %s, %s, %s, %s, %s)" % (self.parent.id, self.condition, self.child.id, self.acc_frontier, self.finite, self.infinite)
        # return "Edge(%s, %s, %s)" % (self.parent.id, self.condition, self.child.id)
    
    def truth(self, state):
        return self.condition.truth([state])
        

class AutomatonState(object):
    def __init__(self, id, accepting_variables={}):
        self.id = id
        self.edges = []
        self.parents = []
        self.accepting_vars = set()
        self.finite = set()
        self.infinite = set()
    
    def set_accepting_vars(self, accepting_vars):
        self.accepting_vars = accepting_vars
    
    def set_accepting(self, i, fin_or_inf):
        if fin_or_inf:
            self.infinite.add(i)
        else:
            self.finite.add(i)
    
    def add_parent(self, edge):
        self.parents.append(edge)
    
    def add_edge(self, edge):
        self.edges.append(edge)
        
        # if (edge.child == edge.parent) and edge.truth({}):
        #     self.terminal = True
    
    def __repr__(self) -> str:
        # return "State(%s, %s, %s)" % (self.id, self.finite, self.infinite)
        return "State(%s)" % (self.id)

class Automaton(object):  # Now for GENERALIZED LDBA --> LDBA MAPPING
    def __init__(self, formula="True", oa_type='dra', rabinizer=None, autobuild=False) -> None:
        
        formula = formula.replace("~", "!")
        self.formula = self.build_formula(formula, autobuild)
        self.oa_type = oa_type

        env = os.environ.copy()
        rabinizer = rabinizer if rabinizer is not None else './rabinizer4/bin/ltl2dra'
        
        if oa_type == 'dra':
            out=check_output([rabinizer, '-c', self.formula] , shell=platform=='win32', env=env)
        elif oa_type == 'ldba':
            out=check_output([rabinizer, '-d', '-e', self.formula] , shell=platform=='win32', env=env)
            out, n_eps = self.parse_ldba_hoa(out)
        elif oa_type == 'gldba':
            out=check_output([rabinizer, '-e', self.formula] , shell=platform=='win32', env=env)
            out, n_eps = self.parse_ldba_hoa(out)
        filename = self.random_hoa_filename()
        with open(filename,'wb') as f:
            f.write(out)
        
        # out=check_output(['autfilt', '-S', filename] , shell=platform=='win32', env=env)
        # os.remove(filename)
        
        # filename = self.random_hoa_filename()
        # with open(filename,'wb') as f:
        #     f.write(out)
        
        spot.setup()
        aut = spot.automaton(filename)
        # spot_oa.merge_edges()  # For better visualization
        os.remove(filename)
        print('Transition-based Acceptance')
        print(aut.to_str('hoa'))

        if oa_type == 'ldba':
            aut = spot.postprocess(aut, 'sbacc')
            out = self.remove_eps(aut.to_str('hoa'), n_eps)

            filename = self.random_hoa_filename()
            with open(filename,'wb') as f:
                f.write(out)

            spot.setup()
            aut = spot.automaton(filename)
            # spot_oa.merge_edges()  # For better visualization
            os.remove(filename)
            print('State-based Acceptance')
            print(aut.to_str('hoa'))

        # aut = spot.postprocess(aut, 'det', 'complete', 'sbacc')
        # aut = spot.postprocess(aut, 'sbacc')
        # aut = spot.postprocess(aut, 'sbacc')
        self.accepting_states = set()
        
        self.G = nx.DiGraph()
        if oa_type == 'gldba':
            self.parse_generalized(aut)
        else:
            self.parse(aut)
        #import pdb; pdb.set_trace()
    
    def build_formula(self, formula, autobuild):
        if not autobuild:
            return formula
        
        parsed_formula = ""
        for subformula in formula:
            operator, low_, high_, template, base_final, base_initial = subformula#.replace(' ','').split(',')
            if operator == '': 
                parsed_formula += '&' + template
            elif base_initial != '':
                X = [template % x for x in range(int(low_), int(high_))]
                out = X[-1]
                for i, x in enumerate(X[::-1][1:]):
                    out = operator % (x, out)
                parsed_formula += base_final % out
            else:
                parsed_formula += operator.join([template % x for x in range(int(low_), int(high_))])

        return parsed_formula

    def remove_eps(self, out, n_eps):
        header, body = out.split('--BODY--\n')
        new_body = []
        body_lines = body.splitlines()  # Ignore the last line

        # Get the number of states
        n_qs = 0  # The number of states
        for line in reversed(body_lines):  # Loop over all states because the states might not be ordered.
            if line.startswith('State'):
                n_qs = max(int(line[7:].split(' ')[0]),n_qs)  # Get the maximum of them

        n_qs += 2  # +1 because the index origin is 0 and +1 for the trap state


        split_header = header.split('\n')

        n_states = split_header[1].split(': ')
        n_states[1] = str(n_qs)
        split_header[1] = ': '.join(n_states)

        AP = split_header[3].split(' \"eps')[0]
        # new_AP = AP + ' ' + ' '.join(["\"eps%d\"" % x for x in range(1, 1+n_eps)])
        split_new_AP = AP.split(" ")
        n_aps = int(split_new_AP[1])
        split_new_AP[1] = str(int(split_new_AP[1]) - n_eps)
        new_AP = ' '.join(split_new_AP)
        split_header[3] = new_AP
        header = '\n'.join(split_header)

        for line in body_lines[:-1]:
            if any([line.startswith('[%d' % x) for x in range(n_aps-n_eps, n_aps)]):
                continue
        
            new_body.append(line)
        
        new_body.append('State: %d' % (n_qs-1))
        new_body.append('[t] %d' % (n_qs-1))
        
        new_body.append(body_lines[-1])
        new_body = '\n'.join(new_body)
        new_out = '--BODY--\n'.join([header,new_body]) + '\n'
        return new_out.encode('utf-8')
            
    def powerset(self, a):
        """Returns the power set of the given list.

        Parameters
        ----------
        a : list
            The input list.

        Returns
        -------
        out: str
            The power set of the list.
        """
        return chain.from_iterable(combinations(a, k) for k in range(len(a)+1))
    
    def parse_ldba_hoa(self, out):
        # Remove epsilons first
        n_eps = 0
        self.eps = {}
        out = out.decode('utf-8')
        print(out)
        header, body = out.split('--BODY--\n')
        new_body = []
        body_lines = body.splitlines()  # Ignore the last line

        n_aps = int(header.split('\n')[-2].split(" ")[1])

        # Get the number of states
        n_qs = 0  # The number of states
        for line in reversed(body_lines):  # Loop over all states because the states might not be ordered.
            if line.startswith('State'):
                n_qs = max(int(line[7:]),n_qs)  # Get the maximum of them

        n_qs += 2  # +1 because the index origin is 0 and +1 for the trap state

        count = -1
        for line in body_lines[:-1]:
            if line.startswith('State'):
                
                if count == 0:
                    new_body.append('[t] %d' % (n_qs-1))
                
                q = int(line[7:])  # Update the state to be parsed
                new_body.append(line)
                count = 0
                
            else:
                # Parse the transition into three parts
                _, _label, _dst, _, _acc_set = re.findall('(\[(.*)\])? ?(\d+) ?(\{(.*)\})?',line)[0]
                dst = int(_dst)  # Get the destination
                
                if not _label:  # If there is no label then the transition is an epsilon-move
                    if q not in self.eps:
                        self.eps[q] = []
                    self.eps[q].append(dst)

                    count += 1
                    new_body.append('[%d] %s' % (n_aps + len(self.eps[q]) - 1, dst)) # to be removed after trans->state based
                    n_eps = max(len(self.eps[q]), n_eps)
                else:
                    count += 1
                    new_body.append(line)

        new_body.append('State: %d' % (n_qs-1))
        new_body.append('[t] %d' % (n_qs-1))

        new_body.append(body_lines[-1])
        new_body = '\n'.join(new_body)
        
        
        
        split_header = header.split('\n')
        AP = split_header[-2]
        new_AP = AP + ' ' + ' '.join(["\"eps%d\"" % x for x in range(1, 1+n_eps)])
        split_new_AP = new_AP.split(" ")
        split_new_AP[1] = str(int(split_new_AP[1]) + n_eps)
        new_AP = ' '.join(split_new_AP)
        split_header[-2] = new_AP
        header = '\n'.join(split_header)
        
        new_out = '--BODY--\n'.join([header,new_body]) + '\n'
        return new_out.encode('utf-8'), n_eps
    
    def random_hoa_filename(self):
        """Returns a random file name.

        Returns
        -------
        filename: str
            A random file name.
        """
        filename = 'temp_%032x.hoa' % random.getrandbits(128)
        while os.path.isfile(filename):
            filename = 'temp_%032x.hoa' % random.getrandbits(128)
        return filename
    
    def state_to_index(self, buchi_state, frontier_state):
        return buchi_state * len(self.all_frontier_states) + self.all_frontier_states.index(frontier_state)
    
    def index_to_state(self, index):
        n_columns = self.num_frontier_states
        return index // n_columns, self.all_frontier_states[index % n_columns]
    
    def parse(self, atm:Union[spot.twa, spot.twa_graph], additional_conditions=[]):

        parser = LTLfParser()

        self.states = dict()  # type: Dict[int, Node]
        acceptance_type = atm.get_acceptance()
        # disjuncts = acceptance_type.top_disjuncts()
        # acceptance_sets = np.array([[[x for x in disjunct.top_conjuncts()[0].fin_unit().sets()][0], [x for x in  disjunct.top_conjuncts()[1].inf_unit().sets()][0]] for disjunct in disjuncts])

        self.aps = [x.ap_name() for x in atm.ap()]
        # queue = atm.num_states()  # type: Set[int]
        processed = set()                      # type: Set[int]
        for state_num in range(atm.num_states()):
            # state_num = queue.pop()
            processed.add(state_num)

            src = self.states.setdefault(state_num, AutomatonState(state_num))
            for e in list(atm.out(state_num)):  # type: spot.twa_graph_edge_storage
                # if e.dst not in processed:
                #     queue.add(e.dst)

                dst_node = self.states.setdefault(e.dst, AutomatonState(e.dst))
                
                # state-based
                if e.acc.count() > 0: 
                    # acc = [x for x in e.acc.sets()][0]
                    # for (i, fin_or_inf) in np.atleast_2d(np.hstack(np.where(acceptance_sets == acc))).tolist():
                        # src.set_accepting(i, fin_or_inf)
                    self.accepting_states.add(src.id)
                    src.set_accepting(0, 1)

                #trans based
                
                conditions = parse_bdd(e.cond, atm.get_dict())

                for condition, condition_dict in conditions.items():
                    edge = AutomatonEdge(src, parser(condition), condition_dict, dst_node)

                    # # transition-based
                    # if e.acc.count() > 0: 
                    #     # acc = [x for x in e.acc.sets()][0]
                    #     # for (i, fin_or_inf) in np.atleast_2d(np.hstack(np.where(acceptance_sets == acc))).tolist():
                    #     #     edge.set_accepting(i, fin_or_inf)
                    #     # self.accepting_states.add(src)
                    #     edge.set_accepting(0, 1)

                    if all([x.satisfied(edge) for x in additional_conditions]):
                        src.add_edge(edge)
                        dst_node.add_parent(edge)
                    else:
                        print('Edge Deleted: ', edge)
                    self.G.add_edge(src.id, dst_node.id, edge = edge, condition = condition)
                    

        self.n_states = len(self.states)
        self.num_buchi_states = self.n_states
        self.start_state = atm.get_init_state_number()
    
    def parse_generalized(self, atm:Union[spot.twa, spot.twa_graph], additional_conditions=[]):

        parser = LTLfParser()
        #import pdb; pdb.set_trace()
        self.states = dict()  # type: Dict[int, Node]
        self.temp_states = dict()
        acceptance_type = atm.get_acceptance()
        self.frontier_set = eval(atm.get_acceptance().inf_unit().as_string())
        self.all_frontier_states = powerset(self.frontier_set)
        # disjuncts = acceptance_type.top_disjuncts()
        # acceptance_sets = np.array([[[x for x in disjunct.top_conjuncts()[0].fin_unit().sets()][0], [x for x in  disjunct.top_conjuncts()[1].inf_unit().sets()][0]] for disjunct in disjuncts])
        self.num_buchi_states = atm.num_states()
        self.num_frontier_states = len(self.all_frontier_states)
        self.aps = [x.ap_name() for x in atm.ap()]
        # queue = atm.num_states()  # type: Set[int]
        self.buchi_state_to_accept_vars = {}
        processed = set()                      # type: Set[int]
        # add all the nodes in the grid-graph first.
        for state_num in range(atm.num_states()):
            accepting_set = eval(atm.state_acc_sets(state_num).as_string())
            if len(accepting_set) == 0:
                accepting_set = set()
            self.buchi_state_to_accept_vars[state_num] = accepting_set
            for frontier_idx1, frontier_state1 in enumerate(self.all_frontier_states): 
                # state_num = queue.pop()
                new_state_num = self.state_to_index(state_num, frontier_state1)
                processed.add(new_state_num)
                

                src = self.temp_states.setdefault(new_state_num, AutomatonState(new_state_num, {}))
                src.set_accepting_vars(accepting_set)
                    
        # now, add the edges.
        for state_num in range(atm.num_states()):
            for e in list(atm.out(state_num)):  # type: spot.twa_graph_edge_storage
                # if e.dst not in processed:
                #     queue.add(e.dst)
                dst_accept_vars = self.buchi_state_to_accept_vars[e.dst]
                # state-based
        
                for frontier_state1 in self.all_frontier_states:
                    src_node = self.temp_states[self.state_to_index(state_num, frontier_state1)]
                    if not src_node.accepting_vars.issubset(frontier_state1):
                        continue # unreachable node with mismatched frontier state to accepting vars at that node
                    if frontier_state1 == self.frontier_set: # accepting node! reset here.
                        dest_frontier = dst_accept_vars
                    else:
                        dest_frontier = frontier_state1.union(dst_accept_vars)
                    destination_node_idx = self.state_to_index(e.dst, dest_frontier)
                    dst_node = self.temp_states[destination_node_idx]
                        
                    conditions = parse_bdd(e.cond, atm.get_dict())

                    for condition, condition_dict in conditions.items():

                        edge = AutomatonEdge(src_node, parser(condition), condition_dict, dst_node, dest_frontier)

                        if all([x.satisfied(edge) for x in additional_conditions]):
                            src_node.add_edge(edge)
                            dst_node.add_parent(edge)
                        else:
                            print('Edge Deleted: ', edge)
                        #self.G.add_edge(src_node.id, dst_node.id, edge = edge, condition = condition, accepting_vars=dest_frontier)
        # now, MERGE
        for frontier_idx, frontier_st in enumerate(self.all_frontier_states):
            # these will be our nodes.
            self.states[frontier_idx] = AutomatonState(frontier_idx, frontier_st)
            if frontier_st == self.frontier_set:
                self.accepting_states.add(frontier_idx)
        self.states[len(self.all_frontier_states)] = AutomatonState(len(self.all_frontier_states), set())
        self.all_edges = []
        for temp_idx, temp_state in self.temp_states.items():
            temp_buchi, temp_parent_frontier = self.index_to_state(temp_idx)
            temp_parent_frontier_idx = self.all_frontier_states.index(temp_parent_frontier)
            for temp_edge in temp_state.edges:
                temp_child = temp_edge.child
                temp_child_buchi, temp_child_frontier = self.index_to_state(temp_child.id)
                # if temp_child_frontier == temp_parent_frontier:
                #     continue
                # they're in different frontiers, so add this edge between the correct nodes
                temp_child_frontier_idx = self.all_frontier_states.index(temp_child_frontier)
                if (temp_parent_frontier_idx, temp_child_frontier_idx, temp_edge.condition) in self.all_edges:
                    continue
                else: # do this to avoid duplicates
                    self.all_edges.append((temp_parent_frontier_idx, temp_child_frontier_idx, temp_edge.condition))
                if temp_edge.condition == flloat.ltlf.LTLfTrue():
                    continue  #TODO: this is kind of a hack.
                # if temp_parent_frontier_idx == 3 and temp_child_frontier_idx == 0:
                #     import pdb; pdb.set_trace()
                if temp_buchi == atm.num_states() - 1:
                    continue  # ignore transitions out of the last state
                new_edge = AutomatonEdge(self.states[temp_parent_frontier_idx], temp_edge.condition, temp_edge.condition_dict, self.states[temp_child_frontier_idx], temp_edge.acc_frontier)
                self.states[temp_parent_frontier_idx].add_edge(new_edge)
                self.states[temp_child_frontier_idx].add_parent(new_edge)
                self.G.add_edge(self.states[temp_parent_frontier_idx].id, self.states[temp_child_frontier_idx].id, edge = new_edge, condition = new_edge.condition, accepting_vars=new_edge.acc_frontier)
        self.n_states = len(self.states)
        # init_b_state = atm.get_init_state_number()
        #self.start_state = self.state_to_index(init_b_state, set())
        self.start_state = self.all_frontier_states.index(set())
        #import pdb; pdb.set_trace()

class HashableDict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))

class AutomatonRunner(object):
    def __init__(self, automaton) -> None:
        self.automaton = automaton
        self.n_states = self.automaton.n_states
        self.edges_visited = {} #set()
        
        # for LCRL
        self.accepting_sets = self.automaton.accepting_states
        self.epsilon_transitions = self.automaton.eps
        # import pdb; pdb.set_trace()
    
    def is_rejecting(self, state):
        # b_state, f_state = self.automaton.index_to_state(state)
        return state == self.automaton.n_states - 1

    def set_state(self, state):
        # b_state, f_state = self.automaton.index_to_state(state)
        # self.current_frontier_set = f_state
        assert (state >= 0) and (state <= self.n_states), 'Setting Automaton to invalid state'
        self.current_state = state
    
    def get_state(self, one_hot = False):
        if one_hot:
            arr = np.zeros(self.n_states)
            arr[self.current_state] = 1
            return arr
        else:
            return self.current_state
    
    def terminal(self):
        current_state = self.get_state()
        if self.automaton.states[current_state].terminal: return True

        #check if condition has been satisfied up to this point
        next_state = self.step({'alive': 0})
        if (self.automaton.states[next_state].terminal) and (self.automaton.states[next_state].accepting):
            return True
        
        self.set_state(current_state)
        return False

    def reward(self):
        if (self.automaton.states[self.current_state].accepting):
            return 1
        else:
            return -1
    
    def reset(self):
        self.current_state = self.automaton.start_state
        # self.current_frontier_set = {}
    
    def epsilon_step(self, action):
        try:
            current_state = self.automaton.eps[self.current_state][action]
            return current_state, -1
        except:
            import pdb; pdb.set_trace()
            assert 'This epsilon step doesnt exist, (q,e) = (%s, %s)' % (self.current_state, action)
        

    def step(self, state):
        if state is None: state = {}
        state.update({'internal_state': self.current_state})
        dict_hash = hash(HashableDict(state))
        if dict_hash in self.edges_visited: 
            edge = self.edges_visited[dict_hash]
            assert edge.truth(state)
            self.current_state = edge.child.id
            return self.current_state, edge

        for edge in self.edges():
            if edge.truth(state):
                self.edges_visited[dict_hash] = edge 
                self.current_state = edge.child.id
                return self.current_state, edge
        
        # else: sink
        # self.edges_visited[dict_hash] = edge #self.edges_visited.add(edge)
        self.current_state = self.n_states - 1
        # self.current_frontier_set = {}
        return self.current_state, None

    def edges(self, start=None, end=None):
        if (start is None) and (end is None):
            return self.automaton.states[self.current_state].edges
        if start is not None:
            if end is not None:
                return [x for x in self.automaton.states[start].edges if x.child.id == end]
            else:
                return self.automaton.states[start].edges