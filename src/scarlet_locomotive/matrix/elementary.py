import torch

#Perform elementary operation R1 ↔ R2 using rowswap function
def SwapRow (M, src, tgt):
  
  M = M.clone()
  M[[src, tgt]] = M[[tgt, src]]
  return M

#Perform elementary operation 1/3 R1 using rowscale function on the resulting matrix from the previous step.
def ScaleRow(M, row, fraction):

  M = M.clone()
  M[row] = M[row] * fraction
  return M

#Perform elementary operation R3 = −3R1 + R3 using rowreplacement function on the resulting matrix from the previous step.
def ReplacementRow(M, i, j, ScaleJ, ScaleK):

  M = M.clone()
  
  ScaledI = ScaleRow(M, i, ScaleJ)[i]
  ScaledJ = ScaleRow(M, j, ScaleK)[j]

  #R1 + R3
  M[i] = ScaledI + ScaledJ
  return M

def ReducedRowEchelonForm(M):
    M = M.clone()
    NumberRows, NumberColumns = M.shape
    PivotRow = 0

    for col in range(NumberColumns):
        if PivotRow >= NumberRows:
            break

        #Find rows at or below PivotRow with non-zero entries in this column
        candidates = []
        for r in range(PivotRow, NumberRows):
            if torch.abs(M[r, col]) > 1e-6:
                candidates.append(r)
        
        # If column is all zeros, move to next column
        if not candidates:
            continue

        #Swap candidate row into the PivotRow position
        target = candidates[0]
        if target != PivotRow:
            M = SwapRow(M, PivotRow, target)

        #Scale the pivot row so the pivot entry becomes 1
        pivot_value = M[PivotRow, col].item()
        M = ScaleRow(M, PivotRow, 1.0 / pivot_value)

        #Eliminate entries in ALL other rows
        for r in range(NumberRows):
            if r != PivotRow and torch.abs(M[r, col]) > 1e-6:
                factor = -M[r, col].item()
                M = ReplacementRow(M, r, PivotRow, 1.0, factor)

        #Advance to the next pivot row
        PivotRow += 1

    #Clean up small negative zeros
    M[torch.abs(M) < 1e-6] = 0.0
    return M
