// Programmable ternary comparator/add-subtract baseline.
module programmable_ternary_dot_16(
    input  logic [31:0] weights,
    input  logic [31:0] activations,
    output logic signed [5:0] result
);
    function automatic logic signed [5:0] ternary_product(
        input logic [1:0] weight,
        input logic [1:0] activation
    );
        case ({weight, activation})
            4'b0101, 4'b1010: ternary_product = 6'sd1;
            4'b0110, 4'b1001: ternary_product = -6'sd1;
            default:          ternary_product = '0;
        endcase
    endfunction
    always_comb begin
        result = ((((ternary_product(weights[0 +: 2], activations[0 +: 2]) + ternary_product(weights[2 +: 2], activations[2 +: 2])) + (ternary_product(weights[4 +: 2], activations[4 +: 2]) + ternary_product(weights[6 +: 2], activations[6 +: 2]))) + ((ternary_product(weights[8 +: 2], activations[8 +: 2]) + ternary_product(weights[10 +: 2], activations[10 +: 2])) + (ternary_product(weights[12 +: 2], activations[12 +: 2]) + ternary_product(weights[14 +: 2], activations[14 +: 2])))) + (((ternary_product(weights[16 +: 2], activations[16 +: 2]) + ternary_product(weights[18 +: 2], activations[18 +: 2])) + (ternary_product(weights[20 +: 2], activations[20 +: 2]) + ternary_product(weights[22 +: 2], activations[22 +: 2]))) + ((ternary_product(weights[24 +: 2], activations[24 +: 2]) + ternary_product(weights[26 +: 2], activations[26 +: 2])) + (ternary_product(weights[28 +: 2], activations[28 +: 2]) + ternary_product(weights[30 +: 2], activations[30 +: 2])))));
    end
endmodule
