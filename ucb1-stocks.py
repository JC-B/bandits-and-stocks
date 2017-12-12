from ucb1 import ucb1
from stats import stats
from stocks import *
from random import shuffle

def ucb1Stocks(stockTable):
   tickers = list(stockTable.keys())
   shuffle(tickers) # note that this makes the algorithm SO unstable
   numRounds = len(stockTable[tickers[0]])
   numActions = len(tickers)

   reward = lambda choice, t: payoff(stockTable, t, tickers[choice])
   singleActionReward = lambda j: sum([reward(j,t) for t in range(numRounds)])

   bestAction = max(range(numActions), key=singleActionReward)
   bestActionCumulativeReward = singleActionReward(bestAction)

   cumulativeReward = 0
   t = 0
   ucb1Generator = ucb1(numActions, reward)
   for (chosenAction, reward, ucbs) in ucb1Generator:
      cumulativeReward += reward
      t += 1
      if t == numRounds:
         break

   return cumulativeReward, bestActionCumulativeReward, ucbs, tickers[bestAction]


prettyList = lambda L: ', '.join(['%.3f' % x for x in L])
payoffStats = lambda data: stats(ucb1Stocks(data)[0] for _ in range(5000))


def runExperiment(table):
   print("(Expected payoff, variance) over 5000 trials is %r" % (prettyList(payoffStats(table)),))
   reward, bestActionReward, ucbs, bestStock = ucb1Stocks(table)
   print("For a single run: ")
   print("Payoff was %.2f" % reward)
   print("Regret was %.2f" % (bestActionReward - reward))
   print("Best stock was %s at %.2f" % (bestStock, bestActionReward))
   print("ucbs: %r" % prettyList(ucbs))


if __name__ == "__main__":
   table = readInStockTable('stocks/fortune-500.csv')
   print("----F500----")
   runExperiment(table)
   payoffGraph(table, list(sorted(table.keys())), cumulative=True, imgfile='images/ucb1-f500-rewards.png')

   print()

   table2 = readInStockTable('stocks/random-stocks.csv')
   print("----Random Stocks----")    
   runExperiment(table2)
   payoffGraph(table2, list(sorted(table2.keys())), cumulative=True, imgfile='images/ucb1-random-stocks-rewards.png')

