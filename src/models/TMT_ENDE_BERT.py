import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertModel
from src.models.TMT_modeling_BERT import TMTBertModel, TMTBertLMHeadModel
import copy

class TMT(nn.Module):
    def __init__(self, args, num_sp_unit, num_txt_unit, num_im_unit):
        super().__init__()
         
        print(f"Num_txt_Token: {num_txt_unit}\nNum_speech_Token: {num_sp_unit}\nNum_image_Token: {num_im_unit}")

        bert_model = BertModel.from_pretrained("bert-base-uncased")
        encoder_config = bert_model.config
        encoder_config.update({'txt': args.txt, 'sp': args.sp, 'im': args.im, 'vocab_size': num_txt_unit, 'sp_vocab_size': num_sp_unit, 'im_vocab_size': num_im_unit, 'initializer_range': 0.01})
        encoder = TMTBertModel(encoder_config, add_pooling_layer=False)
        decoder_config = copy.copy(encoder_config)
        decoder_config.update({'is_decoder': True, 'add_cross_attention': True})
        decoder = TMTBertLMHeadModel(decoder_config)

        encoder.load_state_dict(bert_model.state_dict(), strict=False)
        decoder.bert.load_state_dict(bert_model.state_dict(), strict=False)
        del bert_model
    
        self.encoder = encoder
        self.decoder = decoder
        
        self.mode = args.mode
        
        self.bos = args.bos
        self.eos = args.eos
        self.pad = args.pad
        
        if self.mode == 'test':
            self.beam_size = args.beam_size

        self.gen_max_len = args.gen_max_len

        self.im_sp, self.im_txt, self.sp_txt, self.sp_im, self.txt_sp, self.txt_im = args.im_sp, args.im_txt, args.sp_txt, args.sp_im, args.txt_sp, args.txt_im

    def forward(self, im_unit, sp_unit, txt_unit, sp_unit_len, txt_unit_len):
        # im -> sp
        if self.im_sp:
            output_is = self.forward_task(im_unit.clone(), None, sp_unit.clone(), sp_unit_len, input_modal='image', output_modal='speech', inference=False)
            is_loss = output_is.loss
        else:
            output_is = None
            is_loss = 0

        # im -> txt
        if self.im_txt:
            output_it = self.forward_task(im_unit.clone(), None, txt_unit.clone(), txt_unit_len, input_modal='image', output_modal='text', inference=False)
            it_loss = output_it.loss
        else:
            output_it = None
            it_loss = 0

        # sp -> txt
        if self.sp_txt:
            output_st = self.forward_task(sp_unit.clone(), sp_unit_len, txt_unit.clone(), txt_unit_len, input_modal='speech', output_modal='text', inference=False)
            st_loss = output_st.loss
        else:
            output_st = None
            st_loss = 0

        # txt -> sp
        if self.txt_sp:
            output_ts = self.forward_task(txt_unit.clone(), txt_unit_len, sp_unit.clone(), sp_unit_len, input_modal='text', output_modal='speech', inference=False)
            ts_loss = output_ts.loss
        else:
            output_ts = None
            ts_loss = 0

        # sp -> im
        if self.sp_im:
            output_si = self.forward_task(sp_unit.clone(), sp_unit_len, im_unit.clone(), None, input_modal='speech', output_modal='image', inference=False)
            si_loss = output_si.loss
        else:
            output_si = None
            si_loss = 0
        
        # txt -> im
        if self.txt_im:
            output_ti = self.forward_task(txt_unit.clone(), txt_unit_len, im_unit.clone(), None, input_modal='text', output_modal='image', inference=False)
            ti_loss = output_ti.loss
        else:
            output_ti = None
            ti_loss = 0
        
        return output_is, output_it, output_st, output_si, output_ts, output_ti, is_loss, it_loss, st_loss, si_loss, ts_loss, ti_loss

    def forward_task(self, input_tensor, input_len, target_tensor, target_len, input_modal, output_modal, inference=False):
        max_input_len = input_tensor.size(1)
        if input_len is None:
            input_len = torch.tensor(max_input_len, device=input_tensor.device).repeat(input_tensor.size(0))
        input_mask = self.generate_mask(input_len, max_input_len).cuda()

        if not inference:
            input_ids = target_tensor.clone()
            target_tensor[target_tensor == self.pad] = torch.tensor(-100).cuda()
            max_tgt_len = target_tensor.size(1)
            if target_len is None:
                target_len = torch.tensor(max_tgt_len, device=target_tensor.device).repeat(target_tensor.size(0))

            tgt_key_mask = self.generate_mask(target_len, max_tgt_len).cuda()

            encoder_hidden_states = self.encoder(
                input_ids=input_tensor,
                attention_mask=input_mask,
                input_modal=input_modal,
            ).last_hidden_state

            output = self.decoder(
                input_ids=input_ids,
                labels=target_tensor,
                attention_mask=tgt_key_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=input_mask,
                output_modal=output_modal,
            )

        else:
            encoder_hidden_states = self.encoder(
                input_ids=input_tensor,
                attention_mask=input_mask,
                input_modal=input_modal,
            ).last_hidden_state

            output = self.decoder.generate(
                input_ids=None,
                output_modal=output_modal, 
                max_length=33 if output_modal == 'image' else (self.gen_max_len if self.mode == 'test' else 128),
                bos_token_id=self.bos,
                eos_token_id=self.eos,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=input_mask,
                num_beams=self.beam_size if self.mode == 'test' else 1,
            )
            
        return output
    
    def generate_mask(self, length, sz):
        masks = []
        for i in range(length.size(0)):
            mask = [1] * length[i]
            mask += [0] * (sz - length[i])
            masks += [torch.tensor(mask)]
        masks = torch.stack(masks, dim=0)
        return masks    #1~~~0

    def generate_input_mask(self, length, sz):
        masks = []
        for i in range(length.size(0)):
            mask = [1] * length[i]
            mask += [0] * (sz - length[i])
            masks += [torch.tensor(mask)]
        masks = torch.stack(masks, dim=0).bool()
        return ~masks   #False~~~True