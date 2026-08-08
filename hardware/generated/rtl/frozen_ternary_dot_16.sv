// Generated from blocks.0.attention.qkv.weight row 0, offset 192.
// Ternary encoding: 2'b00=0, 2'b01=+1, 2'b10=-1, 2'b11=reserved.
module frozen_ternary_dot_16(
    input  logic [31:0] activations,
    output logic signed [5:0] result
);
    localparam logic [31:0] WEIGHTS = 32'b00001001010001010010100010010101;
    function automatic logic signed [5:0] decode_activation(input logic [1:0] code);
        case (code)
            2'b01: decode_activation = 6'sd1;
            2'b10: decode_activation = -6'sd1;
            default: decode_activation = '0;
        endcase
    endfunction
    always_comb begin
        result = ((((decode_activation(activations[0 +: 2]) + decode_activation(activations[2 +: 2])) + (decode_activation(activations[4 +: 2]) + -decode_activation(activations[6 +: 2]))) + (('0 + -decode_activation(activations[10 +: 2])) + (-decode_activation(activations[12 +: 2]) + '0))) + (((decode_activation(activations[16 +: 2]) + decode_activation(activations[18 +: 2])) + ('0 + decode_activation(activations[22 +: 2]))) + ((decode_activation(activations[24 +: 2]) + -decode_activation(activations[26 +: 2])) + ('0 + '0))));
    end
endmodule
