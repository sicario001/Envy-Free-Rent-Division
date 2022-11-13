import math
from scipy.optimize import linear_sum_assignment
import numpy as np
from docplex.mp.model import Model
import sys

# epsilon = sys.float_info.epsilon
epsilon = 0.00001

class RentDivisionInstance:
    def __init__(self, valuations: list[list[float]], price: float):
        # a 2D list where valuations[i][j] represents the valuation for room j wrt agent i
        self.valuations = valuations
        self.num_agents = len(valuations)
        self.num_rooms = self.num_agents
        self.price = price
        for val in valuations:
            assert self.num_agents == len(val)
            assert math.isclose(sum(val), price)


class RentDivisionAllocation:
    def __init__(self, assignment: list[int], prices: list[float], valuations=None):
        self.num_agents = len(assignment)
        self.num_rooms = len(assignment)
        # assignment[i] -> room that agent i receives
        self.assignment = assignment
        # price[i] -> price of room i
        self.prices = prices
        self.valuations = valuations

    def __str__(self):
        # convert room assignment to upper case letters
        # assignment = [chr(ord('A') + i) for i in self.assignment]
        out = ""
        for i in range(len(self.assignment)):
            out += "Agent " + str(i+1) + " -> Room " + chr(ord('A') + i) + " at price " + str(self.prices[i]) + "\n"
        # out = str(self.assignment) + "\n" + str(self.prices)
        return out

    def get_utilities(self):
        assert self.valuations is not None
        utilities = [
            [(val[i] - self.prices[i]) for i in range(len(val))]
            for val in self.valuations
        ]
        return utilities


def WelfareMaximizingAssignment(instance: RentDivisionInstance) -> list[int]:
    cost = [[-x for x in val] for val in instance.valuations]
    cost = np.array(cost)
    _, col_ind = linear_sum_assignment(cost)
    assignment = list(col_ind)
    return assignment


class RentDivisionAlgorithm:
    @staticmethod
    def solve(instance: RentDivisionInstance) -> RentDivisionAllocation:
        assignment = [i for i in range(instance.num_agents)]
        prices = [
            instance.price / instance.num_rooms for i in range(instance.num_rooms)
        ]
        allocation = RentDivisionAllocation(assignment=assignment, prices=prices)
        return allocation


class EnvyFree(RentDivisionAlgorithm):
    @staticmethod
    def solve(instance: RentDivisionInstance) -> RentDivisionAllocation:
        model = Model(name="envy-free-rent-division")
        p = model.continuous_var_list(keys=instance.num_agents, lb=0.0, name="p")

        # a welfare maximizing assignment
        assignment = WelfareMaximizingAssignment(instance)

        # envy-freeness constraints
        for i in range(instance.num_agents):
            for j in range(instance.num_agents):
                model.add_constraint(
                    instance.valuations[i][assignment[i]] - p[assignment[i]]
                    >= instance.valuations[i][j] - p[j]
                )
        # sum of prices are fixed
        model.add_constraint(
            sum(p[i] for i in range(instance.num_agents)) == instance.price
        )
        solution = model.solve()
        prices = [solution[p[i]] for i in range(instance.num_rooms)]
        allocation = RentDivisionAllocation(
            assignment=assignment, prices=prices, valuations=instance.valuations
        )
        return allocation


class Maximin(RentDivisionAlgorithm):
    @staticmethod
    def solve(instance: RentDivisionInstance) -> RentDivisionAllocation:
        model = Model(name="Maximin-rent-division")
        p = model.continuous_var_list(keys=instance.num_agents, lb=0.0, name="p")

        # R represents the minimum utility
        R = model.continuous_var(name="R")

        # a welfare maximizing assignment
        assignment = WelfareMaximizingAssignment(instance)

        # envy-freeness constraints
        for i in range(instance.num_agents):
            for j in range(instance.num_agents):
                model.add_constraint(
                    instance.valuations[i][assignment[i]] - p[assignment[i]]
                    >= instance.valuations[i][j] - p[j]
                )

        # maximin constraint
        for i in range(instance.num_agents):
            model.add_constraint(
                R <= instance.valuations[i][assignment[i]] - p[assignment[i]]
            )

        # sum of prices are fixed
        model.add_constraint(
            sum(p[i] for i in range(instance.num_agents)) == instance.price
        )
        # maximize R (i.e maximize the minimum utility)
        model.set_objective("max", R)
        solution = model.solve()
        prices = [solution[p[i]] for i in range(instance.num_rooms)]
        allocation = RentDivisionAllocation(
            assignment=assignment, prices=prices, valuations=instance.valuations
        )
        return allocation


class Maxislack(RentDivisionAlgorithm):
    @staticmethod
    def solve(instance: RentDivisionInstance) -> RentDivisionAllocation:
        model = Model(name="Maxislack-rent-division")
        p = model.continuous_var_list(keys=instance.num_agents, lb=0.0, name="p")

        # S represents the minimum slack
        S = model.continuous_var(name="S")

        # a welfare maximizing assignment
        assignment = WelfareMaximizingAssignment(instance)

        # maxislack constraint
        for i in range(instance.num_agents):
            for j in range(instance.num_agents):
                if j != assignment[i]:
                    model.add_constraint(
                        S
                        <= (instance.valuations[i][assignment[i]] - p[assignment[i]])
                        - (instance.valuations[i][j] - p[j])
                    )

        # sum of prices are fixed
        model.add_constraint(
            sum(p[i] for i in range(instance.num_agents)) == instance.price
        )
        # maximize S (i.e maximize the minimum slack)
        model.set_objective("max", S)
        solution = model.solve()
        prices = [solution[p[i]] for i in range(instance.num_rooms)]
        allocation = RentDivisionAllocation(
            assignment=assignment, prices=prices, valuations=instance.valuations
        )
        return allocation


class Lexislack(RentDivisionAlgorithm):
    @staticmethod
    def get_L(fixed_deltas, non_fixed_deltas, instance, assignment):
        model = Model()
        p = model.continuous_var_list(keys=instance.num_agents, lb=0.0, name="p")

        # S represents the minimum slack over non_fixed_deltas
        S = model.continuous_var(name="S")

        # constraints for maximizing minimum of non_fixed_deltas
        for i, j in non_fixed_deltas:
            model.add_constraint(
                S
                <= (instance.valuations[i][assignment[i]] - p[assignment[i]])
                - (instance.valuations[i][j] - p[j])
            )

        # constraints for fixed_deltas
        for i, j in fixed_deltas.keys():
            model.add_constraint(
                (instance.valuations[i][assignment[i]] - p[assignment[i]])
                - (instance.valuations[i][j] - p[j])
                == fixed_deltas[(i, j)]
            )

        # sum of prices are fixed
        model.add_constraint(
            sum(p[i] for i in range(instance.num_agents)) == instance.price
        )

        # maximize S (i.e maximize the minimum slack over non_fixed_deltas)
        model.set_objective("max", S)

        solution = model.solve()

        assert solution is not None

        L = solution[S]

        return L

    @staticmethod
    def get_solution(fixed_deltas, instance, assignment):
        model = Model()
        p = model.continuous_var_list(keys=instance.num_agents, lb=0.0, name="p")

        # constraints for fixed_deltas
        for i, j in fixed_deltas.keys():
            model.add_constraint(
                (instance.valuations[i][assignment[i]] - p[assignment[i]])
                - (instance.valuations[i][j] - p[j])
                == fixed_deltas[(i, j)]
            )

        # sum of prices are fixed
        model.add_constraint(
            sum(p[i] for i in range(instance.num_agents)) == instance.price
        )

        solution = model.solve()
        prices = [solution[p[i]] for i in range(instance.num_rooms)]

        return prices

    @staticmethod
    def check_valid(L, i1, j1, fixed_deltas, non_fixed_deltas, instance, assignment):
        """Check if delta_i1_j1 can be larger than L for a lexislack allocation"""

        model = Model()
        p = model.continuous_var_list(keys=instance.num_agents, lb=0.0, name="p")

        # constraints for fixed_deltas
        for i, j in fixed_deltas.keys():
            model.add_constraint(
                (instance.valuations[i][assignment[i]] - p[assignment[i]])
                - (instance.valuations[i][j] - p[j])
                == fixed_deltas[(i, j)]
            )

        # constraints for non_fixed_deltas
        for i, j in non_fixed_deltas:
            if (i, j) != (i1, j1):
                model.add_constraint(
                    (instance.valuations[i][assignment[i]] - p[assignment[i]])
                    - (instance.valuations[i][j] - p[j])
                    >= L
                )
            else:
                model.add_constraint(
                    (instance.valuations[i1][assignment[i1]] - p[assignment[i1]])
                    - (instance.valuations[i1][j1] - p[j1])
                    >= L + epsilon 
                )

        # sum of prices are fixed
        model.add_constraint(
            sum(p[i] for i in range(instance.num_agents)) == instance.price
        )

        return model.solve() is not None

    @staticmethod
    def solve(instance: RentDivisionInstance) -> RentDivisionAllocation:
        # a welfare maximizing assignment
        assignment = WelfareMaximizingAssignment(instance)

        fixed_deltas = {}
        non_fixed_deltas = set()
        for i in range(instance.num_agents):
            for j in range(instance.num_agents):
                if j != assignment[i]:
                    non_fixed_deltas.add((i, j))

        while len(non_fixed_deltas) > 0:
            L = Lexislack.get_L(fixed_deltas, non_fixed_deltas, instance, assignment)
            flag = False
            for i1, j1 in non_fixed_deltas:
                # Check if delta_i1_j1 can be larger than L for a lexislack allocation
                if not Lexislack.check_valid(
                    L, i1, j1, fixed_deltas, non_fixed_deltas, instance, assignment
                ):
                    flag = True
                    fixed_deltas[(i1, j1)] = L
                    non_fixed_deltas.remove((i1, j1))
                    break
            assert flag
        prices = Lexislack.get_solution(fixed_deltas, instance, assignment)
        allocation = RentDivisionAllocation(
            assignment=assignment, prices=prices, valuations=instance.valuations
        )
        return allocation

class Minimax(RentDivisionAlgorithm):
    @staticmethod
    def solve(instance: RentDivisionInstance) -> RentDivisionAllocation:
        model = Model(name="Minimax-rent-division")
        p = model.continuous_var_list(keys=instance.num_agents, lb=0.0, name="p")

        # R represents the maximum utility
        R = model.continuous_var(name="R")

        # a welfare maximizing assignment
        assignment = WelfareMaximizingAssignment(instance)

        # envy-freeness constraints
        for i in range(instance.num_agents):
            for j in range(instance.num_agents):
                model.add_constraint(
                    instance.valuations[i][assignment[i]] - p[assignment[i]]
                    >= instance.valuations[i][j] - p[j]
                )

        # minimax constraint
        for i in range(instance.num_agents):
            model.add_constraint(
                R >= instance.valuations[i][assignment[i]] - p[assignment[i]]
            )

        # sum of prices are fixed
        model.add_constraint(
            sum(p[i] for i in range(instance.num_agents)) == instance.price
        )
        # minimize R (i.e minimize the maximum utility)
        model.set_objective("min", R)
        solution = model.solve()
        prices = [solution[p[i]] for i in range(instance.num_rooms)]
        allocation = RentDivisionAllocation(
            assignment=assignment, prices=prices, valuations=instance.valuations
        )
        return allocation

def generate_random_valuations(n, price):
    valuations = [[np.random.random() for i in range(n)] for j in range(n)]
    for val in valuations:
        sum_val = sum(val)
        for i in range(n):
            val[i] = val[i]*price/sum_val
        print(sum(val))
    return valuations

def main():
    # valuations = [[4.0, 1.0, 3.0], [2.0, 0.0, 6.0], [3.0, 3.0, 2.0]]
    # price = 8.0
    # price = 10
    # valuations = generate_random_valuations(4, price)
    # print(valuations)

    # instance = RentDivisionInstance(valuations=valuations, price=price)
    # allocation = Maximin.solve(instance=instance)
    # print(allocation)
    # print(allocation.get_utilities())

    # prakhar, sayam
    valuations = [[4000,6000],[7500,2500]]
    print("the valuations are:")
    for i in range(len(valuations)):
        print("Agent "+str(i+1)+": "+str(valuations[i]))
    
    price = 10000
    instance = RentDivisionInstance(valuations=valuations, price=price)

    # Envy-free allocation
    allocation = EnvyFree.solve(instance=instance)
    print("Envy-free allocation :")
    print(allocation)
    # print(allocation.get_utilities())

    allocation = Maximin.solve(instance=instance)
    print("Maximin allocation :")
    print(allocation)
    # print(allocation.get_utilities())

    allocation = Maxislack.solve(instance=instance)
    print("Maxislack allocation :")
    print(allocation)
    # print(allocation.get_utilities())

    allocation = Lexislack.solve(instance=instance)
    print("Lexislack allocation :")
    print(allocation)
    # print(allocation.get_utilities())



main()
