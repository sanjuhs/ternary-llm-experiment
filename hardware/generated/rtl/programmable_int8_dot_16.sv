// Programmable signed INT8 multiply-accumulate baseline.
module programmable_int8_dot_16(
    input  logic signed [127:0] weights,
    input  logic signed [127:0] activations,
    output logic signed [20:0] result
);
    function automatic logic signed [20:0] int8_product(
        input logic signed [7:0] weight,
        input logic signed [7:0] activation
    );
        int8_product = weight * activation;
    endfunction
    always_comb begin
        result = ((((int8_product(weights[0 +: 8], activations[0 +: 8]) + int8_product(weights[8 +: 8], activations[8 +: 8])) + (int8_product(weights[16 +: 8], activations[16 +: 8]) + int8_product(weights[24 +: 8], activations[24 +: 8]))) + ((int8_product(weights[32 +: 8], activations[32 +: 8]) + int8_product(weights[40 +: 8], activations[40 +: 8])) + (int8_product(weights[48 +: 8], activations[48 +: 8]) + int8_product(weights[56 +: 8], activations[56 +: 8])))) + (((int8_product(weights[64 +: 8], activations[64 +: 8]) + int8_product(weights[72 +: 8], activations[72 +: 8])) + (int8_product(weights[80 +: 8], activations[80 +: 8]) + int8_product(weights[88 +: 8], activations[88 +: 8]))) + ((int8_product(weights[96 +: 8], activations[96 +: 8]) + int8_product(weights[104 +: 8], activations[104 +: 8])) + (int8_product(weights[112 +: 8], activations[112 +: 8]) + int8_product(weights[120 +: 8], activations[120 +: 8])))));
    end
endmodule
