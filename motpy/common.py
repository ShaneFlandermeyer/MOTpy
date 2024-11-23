def nextpow2(n: int):
  return 2**(n - 1).bit_length()