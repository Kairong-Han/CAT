import torch

def compute_attention_constraint_loss_batch(mask, attention_map, alpha, beta=10,Amp=1,print_detail=False,loss_type='div'):

    assert mask.shape == attention_map.shape, ""
    batch_size, strlen, _ = mask.shape
    device = attention_map.device
    
    lower_triangle_matrix = torch.tril(torch.ones(strlen,strlen))
   
    total_loss = torch.tensor(0.0, device=device)

    valid_rows = 0
    for i in range(batch_size):
       
        current_mask = mask[i]
        current_attention = attention_map[i]

        for j in range(strlen):
            current_row_mask = current_mask[j]
            current_row_attention = current_attention[j]
            
            if current_row_mask.sum() == 0 or current_row_mask.sum() == current_row_mask.numel():
                continue  

            attention_masked_1 = current_row_attention[(current_row_mask == 1) & (lower_triangle_matrix[j]==1)]
            attention_masked_0 = current_row_attention[(current_row_mask == 0) & (lower_triangle_matrix[j]==1)]

            if attention_masked_1.numel() == 0 or attention_masked_0.numel() == 0:
                continue
            avg_attention_1 = attention_masked_1.mean()
            avg_attention_0 = attention_masked_0.mean()

            ratio = avg_attention_1 / avg_attention_0
            if ratio < alpha:
                if loss_type == 'div':
                    constraint_loss = -torch.log(1+avg_attention_1)/torch.log(1+avg_attention_0)
                elif loss_type == 'div2':
                    constraint_loss = -torch.log(1+avg_attention_1/(1+avg_attention_0))
                elif loss_type == 'nll':
                    constraint_loss = - avg_attention_1/(avg_attention_1+avg_attention_0)
                else:
                    constraint_loss = torch.maximum(alpha - ratio, torch.tensor(0.0, device=device))
            else:
                constraint_loss = torch.tensor(0.0, device=device)
            total_loss += constraint_loss
            valid_rows += 1  # 有效行数累加
           
            if print_detail:
                print(f"Sample {i}, Row {j} - Ratio: {ratio}, Loss: {constraint_loss}")

    if valid_rows > 0:
        total_loss = total_loss / valid_rows
        print(f"total Loss: {total_loss}")

    return Amp * total_loss



def compute_attention_constraint_loss_batch_pinghua(mask, attention_map, alpha, beta=10,Amp=1,print_detail=False,loss_type='div'):

    assert mask.shape == attention_map.shape, ""
    batch_size, strlen, _ = mask.shape
    device = attention_map.device

    lower_triangle_matrix = torch.tril(torch.ones(strlen,strlen))

    total_loss = torch.tensor(0.0, device=device)
    valid_item = 0

    for i in range(batch_size):

        current_mask = mask[i]
        current_attention = attention_map[i]
        total_ratio = 0
        valid_rows = 0
       
         
        for j in range(strlen):
            current_row_mask = current_mask[j]
            current_row_attention = current_attention[j]

            if current_row_mask.sum() == 0 or current_row_mask.sum() == current_row_mask.numel():
                continue  
            attention_masked_1 = current_row_attention[(current_row_mask == 1) & (lower_triangle_matrix[j]==1)]
            attention_masked_0 = current_row_attention[(current_row_mask == 0) & (lower_triangle_matrix[j]==1)]
           
            if attention_masked_1.numel() == 0 or attention_masked_0.numel() == 0:
                continue
            avg_attention_1 = attention_masked_1.mean()
            avg_attention_0 = attention_masked_0.mean()
            ratio = avg_attention_1 / avg_attention_0
            total_ratio += ratio
            # if ratio < alpha:
            #     if loss_type == 'div':
            #         constraint_loss = -torch.log(1+avg_attention_1)/torch.log(1+avg_attention_0)
            #     elif loss_type == 'div2':
            #         constraint_loss = -torch.log(1+avg_attention_1/(1+avg_attention_0))
            #     elif loss_type == 'nll':
            #         constraint_loss = - avg_attention_1/(avg_attention_1+avg_attention_0)
            #     else:
            #         constraint_loss = torch.maximum(alpha - ratio, torch.tensor(0.0, device=device))
            # else:
            #     constraint_loss = torch.tensor(0.0, device=device)
            valid_rows += 1 
        if valid_rows>0:
            total_ratio /= valid_rows

        if total_ratio < alpha and valid_rows > 0:
            valid_item += 1
            if loss_type == 'div':
                constraint_loss = -torch.log(1+avg_attention_1)/torch.log(1+avg_attention_0)
            elif loss_type == 'div2':
                constraint_loss = -torch.log(1+avg_attention_1/(1+avg_attention_0))
            elif loss_type == 'nll':
                constraint_loss = - avg_attention_1/(avg_attention_1+avg_attention_0)
            else:
                constraint_loss = torch.maximum(alpha - total_ratio, torch.tensor(0.0, device=device))
            total_loss += constraint_loss
        else:
            constraint_loss = torch.tensor(0.0, device=device)
            total_loss += constraint_loss
        
            
        if print_detail:
            print(f"Sample {i},  Ratio: {total_ratio}, Loss: {constraint_loss}")

    if valid_item > 0:
        total_loss = total_loss / valid_item
        print(f"total Loss: {total_loss}")
    return Amp * total_loss


def average_attention(attentions):
    
    avg_attention_layers = []
    for layer_attention in attentions:

        avg_attention_heads = layer_attention.mean(dim=1)  # [batch_size, strlen, strlen]

        avg_attention_layers.append(avg_attention_heads)
    
    avg_attention_all_layers = torch.stack(avg_attention_layers, dim=0)  # [num_layers, batch_size, strlen, strlen]
    

    final_avg_attention = avg_attention_all_layers.mean(dim=0)  # [batch_size, strlen, strlen]
    return final_avg_attention

# new_attention = average_attention(outputs.attentions)