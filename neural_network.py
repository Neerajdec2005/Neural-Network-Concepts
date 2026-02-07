_in= [1.3, 3.2, 4.5, 7.2]
weights1= [1.2 ,5.7, 3.1, 2.4]
weights2= [1.7 ,4.5, 3.7, 1.9]
weights3= [4.1 ,2.2, 5.1, 2.9]

bias1=4
bias2=2.4
bias3=3.4

output=[_in[0]*weights1[0] + _in[1]*weights1[1] + _in[2]*weights1[2] + _in[3]*weights1[3] + bias1,
        _in[0]*weights2[0] + _in[1]*weights2[1] + _in[2]*weights2[2] + _in[3]*weights2[3] + bias2,
        _in[0]*weights3[0] + _in[1]*weights3[1] + _in[2]*weights3[2] + _in[3]*weights3[3] + bias3]

print(output)