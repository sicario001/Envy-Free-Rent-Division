from scipy.optimize import linear_sum_assignment
import numpy as np
from docplex.mp.model import Model


class RentDivisionInstance:
    def __init__(self, valuations: list[list[float]], price: float):
        # a 2D list where valuations[i][j] represents the valuation for room j wrt agent i
        self.valuations = valuations
        self.num_agents = len(valuations)
        self.num_rooms = self.num_agents
        self.price = price
        for val in valuations:
            assert self.num_agents == len(val)
            assert sum(val) == price


class RentDivisionAllocation:
    def __init__(self, assignment: list[int], prices: list[float], valuations = None):
        self.num_agents = len(assignment)
        self.num_rooms = len(assignment)
        # assignment[i] -> room that agent i receives
        self.assignment = assignment
        # price[i] -> price of room i
        self.prices = prices
        self.valuations = valuations

    def __str__(self):
        out = str(self.assignment) + "\n" + str(self.prices)
        return out
    
    def get_utilities(self):
        assert self.valuations is not None
        utilities = [[(val[i] - self.prices[i]) for i in range(len(val))] for val in self.valuations]
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
        p = model.continuous_var_list(keys=instance.num_agents, lb=0, name="p")

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
        allocation = RentDivisionAllocation(assignment=assignment, prices=prices, valuations=instance.valuations)
        return allocation


class Maximin(RentDivisionAlgorithm):
    @staticmethod
    def solve(instance: RentDivisionInstance) -> RentDivisionAllocation:
        model = Model(name="envy-free-rent-division")
        p = model.continuous_var_list(keys=instance.num_agents, lb=0, name="p")
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
        allocation = RentDivisionAllocation(assignment=assignment, prices=prices, valuations=instance.valuations)
        return allocation


def main():
    valuations = [[4.0, 1.0, 3.0], [2.0, 0.0, 6.0], [3.0, 3.0, 2.0]]
    price = 8.0
    instance = RentDivisionInstance(valuations=valuations, price=price)
    allocation = Maximin.solve(instance=instance)
    print(allocation)
    print(allocation.get_utilities())



main()
