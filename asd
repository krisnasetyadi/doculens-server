from odoo import models, fields, api, _
from odoo.exceptions import UserError


class ApprovalProductLine(models.Model):
    _inherit = 'approval.product.line'

    def _get_purchase_orders_domain(self, vendor):
        """ Override: always return a domain matching no record, so that
        action_create_purchase_orders() never finds an existing draft PO
        to merge into, and always creates a brand new Purchase Order
        instead - regardless of vendor, currency, or date.
        """
        self.ensure_one()
        return [('id', '=', False)]


class ApprovalRequest(models.Model):
    _inherit = 'approval.request'

    def action_create_purchase_orders(self):
        """ Override total: Bikin PO Baru khusus dari Approval ini, digabung per Seller/Vendor """
        self.ensure_one()

        # 1. Grouping
        seller_lines_map = {}

        lines_to_process = self.product_line_ids.filtered(
            lambda l: not l.purchase_order_line_id
        )

        if not lines_to_process:
            raise UserError(_("Semua item di approval ini sudah memiliki Purchase Order."))

        for line in lines_to_process:
            if not line.seller_id:
                raise UserError(_("Barang %s belum memiliki Seller/Vendor!") % line.product_id.name)

            seller_id = line.seller_id.id
            seller_lines_map.setdefault(seller_id, [])
            seller_lines_map[seller_id].append(line)

        po_ids = []
        PurchaseOrder = self.env['purchase.order']

        # 2. Create PO
        for seller_id, lines in seller_lines_map.items():
            po_line_vals = []
            created_lines = []  

            for line in lines:
                # get price from vendor pricelist (product.supplierinfo),
              
                seller_info = line.product_id.seller_ids.filtered(
                    lambda s: s.partner_id.id == seller_id
                )[:1]
                price_unit = seller_info.price if seller_info else line.product_id.standard_price

                po_line_vals.append((0, 0, {
                    'product_id': line.product_id.id,
                    'name': line.description or line.product_id.name,
                    'product_qty': line.quantity,
                    'product_uom': line.product_uom_id.id,
                    'price_unit': price_unit,
                    'date_planned': fields.Datetime.now(),
                }))
                created_lines.append(line)

            po_vals = {
                'partner_id': seller_id,
                'origin': self.name,
                'order_line': po_line_vals,
            }

            new_po = PurchaseOrder.create(po_vals)
            po_ids.append(new_po.id)

            # 3. back link each approval.product.line to purchase.order.line
            for approval_line, po_line in zip(created_lines, new_po.order_line):
                approval_line.purchase_order_line_id = po_line.id

        # 4. redirect user tp  PO page 
        action = self.env["ir.actions.actions"]._for_xml_id("purchase.purchase_rfq")
        if len(po_ids) == 1:
            action.update({
                'view_mode': 'form',
                'res_id': po_ids[0],
            })
        else:
            action.update({
                'domain': [('id', 'in', po_ids)],
                'view_mode': 'tree,form',
            })
        return action